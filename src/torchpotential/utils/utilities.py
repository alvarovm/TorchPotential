import logging

import numpy as np
import torch

from torchpotential.constants import GPU_INDEX, GPU_MEMORY_LIMIT
from torchpotential.utils import device as _device

log = logging.getLogger(__name__)


def apply_weights(row, fname="w_forces", ename="w_energy"):
    row["tp_atoms"]["_eweights"] = np.array([row[ename]]).reshape(-1, 1)
    row["tp_atoms"]["_fweights"] = np.array(row[fname]).reshape(-1, 1)


def batching_data(df, batch_size=10):
    df.dropna(subset=["tp_atoms"], inplace=True)
    try:
        df.apply(apply_weights, axis=1)
    except Exception:
        pass
    df.index = np.arange(len(df))
    batch_index = chunks(len(df), batch_size)
    batches = []
    for batch in batch_index:
        batch_list = list(df.loc[batch]["tp_atoms"])
        batches.append(supply_batch(batch_list))

    return batches


def check_or_put_unit_weights(dic):
    if "_eweights" not in list(dic):
        dic.update({"_eweights": np.ones([len(dic["_energy"]), 1])})

    if "_fweights" not in list(dic):
        dic.update({"_fweights": np.ones([len(dic["_forces"]), 1])})

    return dic


def chunks(size, chunksize):
    n = max(1, chunksize)
    lst = np.arange(0, size)
    return [lst[i : i + n] for i in range(0, size, n)]


def supply_batch(data):
    batch_data = {
        "ind_i": [],
        "ind_j": [],
        "mu_i": [],
        "mu_j": [],
        "offsets": [],
        "energy": [],
        "forces": [],
        "positions": [],
        "eweights": [],
        "fweights": [],
        "cell": [],
        "cell_map": [],
        "pair_map": [],
        "e_map": [],
    }

    count_atoms = 0
    count_cells = 0
    for j, entry in enumerate(data):
        entry = check_or_put_unit_weights(entry)
        n_atoms = entry["_positions"].shape[0]
        n_pairs = entry["_ind_j"].shape[0]

        batch_data["ind_i"].append(entry["_ind_i"] + count_atoms)
        batch_data["ind_j"].append(entry["_ind_j"] + count_atoms)
        batch_data["mu_i"].append(entry["_mu_i"])
        batch_data["mu_j"].append(entry["_mu_j"])
        batch_data["offsets"].append(entry["_offsets"])
        batch_data["energy"].append(entry["_energy"])
        batch_data["forces"].append(entry["_forces"])
        batch_data["eweights"].append(entry["_eweights"])
        batch_data["fweights"].append(entry["_fweights"])
        batch_data["positions"].append(entry["_positions"])
        batch_data["cell"].append(entry["_cell"])
        batch_data["pair_map"].append([count_cells] * n_pairs)
        batch_data["cell_map"].append([count_cells] * n_atoms)
        batch_data["e_map"].append([j] * n_atoms)

        count_atoms += n_atoms
        count_cells += 1

    batch_data = {k: np.concatenate(v, axis=0) for k, v in batch_data.items()}

    return batch_data


def split_batch(b, max_structures):
    """Split a merged numpy batch (``supply_batch`` output) into pieces of at most ``max_structures``
    structures each, each independently self-contained (its own 0-based atom/cell/pair indexing, as if
    ``supply_batch`` had built it directly from that structure subset).

    Evaluating the pieces and concatenating the results is equivalent to evaluating ``b`` as a whole, but bounds
    the transient device memory of a single ``evaluate()`` call: useful for batches that must not stay resident
    on the device, where "foreign" (not kept) batches of a distributed fit can otherwise need as much transient
    memory as this rank's permanent working set, with no headroom left (see ``FitTorchPotential.predict``).
    ``[b]`` if ``b`` already has at most ``max_structures`` structures, or ``max_structures <= 0``.
    """
    e_map = np.asarray(b["e_map"])
    n_structures = int(e_map.max()) + 1 if len(e_map) else 0
    if max_structures <= 0 or n_structures <= max_structures:
        return [b]
    pair_map = np.asarray(b["pair_map"])
    pieces = []
    for start in range(0, n_structures, max_structures):
        end = min(start + max_structures, n_structures)
        atom_mask = (e_map >= start) & (e_map < end)
        pair_mask = (pair_map >= start) & (pair_map < end)
        atom_offset = int(np.sum(e_map < start))
        pieces.append({
            "ind_i": b["ind_i"][pair_mask] - atom_offset,
            "ind_j": b["ind_j"][pair_mask] - atom_offset,
            "mu_i": b["mu_i"][pair_mask],
            "mu_j": b["mu_j"][pair_mask],
            "offsets": b["offsets"][pair_mask],
            "energy": b["energy"][start:end],
            "forces": b["forces"][atom_mask],
            "positions": b["positions"][atom_mask],
            "eweights": b["eweights"][start:end],
            "fweights": b["fweights"][atom_mask],
            "cell": b["cell"][start:end],
            "cell_map": b["cell_map"][atom_mask] - start,
            "pair_map": pair_map[pair_mask] - start,
            "e_map": e_map[atom_mask] - start,
        })
    return pieces


def batch_errors(b, e, f):
    """Energy/force residuals of one numpy batch: ``(de, de_pa, df, n_atoms_per_structure)``."""
    _, nat = np.unique(b["e_map"], return_counts=True)
    de = b["energy"] - e
    de_pa = de / nat.reshape(-1, 1)
    df = b["forces"] - f
    return de, de_pa, df, nat.reshape(-1, 1)


def get_gpu_memory(gpu_id):
    """Total GPU memory in MB; an integer uses the auto-selected CUDA/XPU backend."""
    if isinstance(gpu_id, int):
        accelerator = _device._available_accelerator()
        if accelerator is None:
            return 0
        gpu_id = torch.device(accelerator, gpu_id)
    device = torch.device(gpu_id)
    if device.type not in ("cuda", "xpu"):
        return 0
    return getattr(torch, device.type).get_device_properties(device).total_memory // (1024 * 1024)


def _set_gpu_config(config=None):
    """Validate ``gpu_config``. ``gpu_ind`` is an int (single device, ``< 0`` = CPU) or a non-empty list of
    non-negative ints (one worker process per entry; repeating an index runs several processes on one GPU)."""
    conf_dict = {GPU_INDEX: 0, GPU_MEMORY_LIMIT: 0}

    if config is not None:
        assert isinstance(config, dict), "gpu_config must be a dict"
        if GPU_INDEX in config:
            idx = config[GPU_INDEX]
            if isinstance(idx, (list, tuple)):
                assert len(idx) > 0, "{} must not be an empty list".format(GPU_INDEX)
                assert all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in idx), (
                    "{} list must contain non-negative integers".format(GPU_INDEX)
                )
                conf_dict[GPU_INDEX] = [int(i) for i in idx]
            else:
                assert isinstance(idx, int), "{} must be an integer or a list of integers".format(GPU_INDEX)
                conf_dict[GPU_INDEX] = idx

        if GPU_MEMORY_LIMIT in config:
            assert isinstance(config[GPU_MEMORY_LIMIT], int), "{} must be an integer number of MB".format(
                GPU_MEMORY_LIMIT
            )
            if config[GPU_MEMORY_LIMIT] < 0:
                raise ValueError("{} must be not negative".format(GPU_MEMORY_LIMIT))
            conf_dict[GPU_MEMORY_LIMIT] = config[GPU_MEMORY_LIMIT]

    return conf_dict


def _physical_card_id(device):
    """Identifier of the physical GPU behind ``device``.

    On a multi-tile Intel XPU node exposed with ``ZE_FLAT_DEVICE_HIERARCHY=FLAT``, each physical card is split
    into several ``xpu:i`` indices (tiles) that torch enumerates consecutively; tiles of the same card share a
    UUID prefix (only its last segment, the tile number, differs). Grouping by this prefix lets
    :func:`warn_on_shared_physical_devices` detect devices that look distinct but are not. Falls back to the
    device index itself (always distinct) for CUDA, CPU, or when no UUID is available (older PyTorch)."""
    device = torch.device(device)
    if device.type == "xpu" and device.index is not None:
        try:
            uuid = getattr(torch.xpu.get_device_properties(device), "uuid", None)
        except Exception:
            uuid = None
        if uuid:
            return "xpu-card:{}".format(str(uuid).rsplit("-", 1)[0])
    return "{}:{}".format(device.type, device.index)


def _all_physical_cards(device_type):
    """The physical cards of every XPU index visible to this process (``device_count()``), used to tell a
    doubled-up tile apart from genuinely wasted hardware. Empty if the accelerator is unavailable."""
    accelerator = getattr(torch, device_type, None)
    if accelerator is None or not getattr(accelerator, "is_available", lambda: False)():
        return set()
    try:
        n = accelerator.device_count()
    except Exception:
        return set()
    return {_physical_card_id(torch.device(device_type, i)) for i in range(n)}


def warn_on_shared_physical_devices(devices):
    """Warns when several ``devices`` double up on a physical GPU while another one visible to this process is
    left entirely unused: a common ``ZE_FLAT_DEVICE_HIERARCHY=FLAT`` pitfall where consecutive indices (e.g.
    ``gpu_ind: [0, 1, 2, 3]``) look like 4 distinct GPUs but are really 2 physical cards (2 tiles each), leaving
    other, equally visible cards idle. Deliberately using every tile of every selected card (including every
    visible tile, as in ``gpu_ind: [0, 1, ..., 11]`` on a fully allocated 12-tile node) is not flagged: nothing
    else visible to this process is left unused in that case.
    """
    xpu_devices = [d for d in devices if torch.device(d).type == "xpu"]
    if len(xpu_devices) < 2:
        return
    selected_cards = {}
    for d in xpu_devices:
        selected_cards.setdefault(_physical_card_id(d), []).append(str(d))
    doubled = {card: names for card, names in selected_cards.items() if len(names) > 1}
    if not doubled:
        return
    unused = _all_physical_cards("xpu") - set(selected_cards)
    if not unused:
        return  # every physical card visible to this process already has a selected tile
    log.warning(
        "%d of the %d selected XPU devices double up on %d physical GPU(s) (%s) while %d other physical GPU(s) "
        "visible to this process are not selected at all. If you intended %d distinct physical GPUs, pick one "
        "tile per card instead (e.g. every other index).",
        sum(len(v) for v in doubled.values()), len(xpu_devices), len(doubled),
        "; ".join("[{}]".format(", ".join(v)) for v in doubled.values()), len(unused), len(xpu_devices),
    )


def get_devices(gpu_config=None):
    """Devices (``torch.device`` list) of the ranks of a fit; the first one is the main process.

    Precedence: ``TORCHPOTENTIAL_DEVICES`` (e.g. ``cuda:0,cuda:1`` or ``xpu:0,xpu:1``), ``TORCHPOTENTIAL_DEVICE``,
    ``gpu_config['gpu_ind']`` (int or list), else ``cuda:0`` / ``xpu:0`` / CPU. Lists use the first available
    accelerator (CUDA, then XPU). More than one device means a multi-process fit.

    On a multi-tile Intel XPU node, consecutive indices can be tiles of the same physical card (see
    :func:`warn_on_shared_physical_devices`); a warning is logged, not raised, since repeating or doubling up
    tiles is also a valid, deliberate choice (testing, or using every tile of a card).
    """
    import os

    env_list = os.environ.get(_device.ENV_DEVICES)
    if env_list:
        devs = [torch.device(d.strip()) for d in env_list.split(",") if d.strip()]
        if devs:
            if len(devs) > 1:
                warn_on_shared_physical_devices(devs)
            return devs
    if os.environ.get(_device.ENV_DEVICE):
        return [torch.device(os.environ[_device.ENV_DEVICE])]

    idx = _set_gpu_config(gpu_config)[GPU_INDEX]
    indices = idx if isinstance(idx, list) else [idx]
    if not isinstance(idx, list) and idx < 0:
        return [torch.device("cpu")]
    accelerator = _device._available_accelerator()
    if accelerator is None:
        if isinstance(idx, list):
            raise RuntimeError("{} {} requested, but neither CUDA nor XPU is available".format(GPU_INDEX, idx))
        return [torch.device("cpu")]
    n = getattr(torch, accelerator).device_count()
    for i in indices:
        assert i < n, "GPU ind {} is requested, but there are only {} {} GPUs".format(i, n, accelerator)
    devices = [torch.device(accelerator, i) for i in indices]
    if len(devices) > 1:
        warn_on_shared_physical_devices(devices)
    return devices


def apply_memory_limit(device, mem_limit):
    """Cap the caching allocator of this process on ``device`` to ``mem_limit`` MB (0 = no limit)."""
    device = torch.device(device)
    if mem_limit and device.type in ("cuda", "xpu"):
        accelerator = getattr(torch, device.type)
        idx = device.index if device.index is not None else accelerator.current_device()
        frac = mem_limit / get_gpu_memory(torch.device(device.type, idx))
        accelerator.set_per_process_memory_fraction(min(frac, 1.0), idx)


def init_gpu_config(gpu_config):
    """Select the compute device of this process. ``gpu_ind < 0`` -> CPU; ``mem_limit`` (MB) caps the caching allocator.

    With a list in ``gpu_ind`` (or ``TORCHPOTENTIAL_DEVICES``) the first entry is the device of this (main) process;
    use :func:`get_devices` for the full list. Environment selection overrides the config's device indices,
    but ``mem_limit`` still applies to every selected GPU, including the main process.
    """
    gpu_config = _set_gpu_config(gpu_config)
    devices = get_devices(gpu_config)
    mem_limit = gpu_config[GPU_MEMORY_LIMIT]
    for device in devices:
        if mem_limit and device.type in ("cuda", "xpu"):
            assert mem_limit <= get_gpu_memory(device), "Requested GPU memory limit is greater than total GPU memory"
    _device.set_device(devices[0])
    if devices[0].type in ("cuda", "xpu"):
        getattr(torch, devices[0].type).set_device(devices[0])
        apply_memory_limit(devices[0], mem_limit)

    return gpu_config
