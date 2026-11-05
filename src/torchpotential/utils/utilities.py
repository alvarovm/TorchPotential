import numpy as np
import torch

from torchpotential.constants import GPU_INDEX, GPU_MEMORY_LIMIT
from torchpotential.utils import device as _device


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


def batch_errors(b, e, f):
    """Energy/force residuals of one numpy batch: ``(de, de_pa, df, n_atoms_per_structure)``."""
    _, nat = np.unique(b["e_map"], return_counts=True)
    de = b["energy"] - e
    de_pa = de / nat.reshape(-1, 1)
    df = b["forces"] - f
    return de, de_pa, df, nat.reshape(-1, 1)


def get_gpu_memory(gpu_id):
    """Total memory of the GPU in MB."""
    return torch.cuda.get_device_properties(gpu_id).total_memory // (1024 * 1024)


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
            indices = conf_dict[GPU_INDEX] if isinstance(conf_dict[GPU_INDEX], list) else [conf_dict[GPU_INDEX]]
            if all(i >= 0 for i in indices) and config[GPU_MEMORY_LIMIT] > 0:
                try:
                    total_gpu_mem = min(get_gpu_memory(i) for i in indices)
                except Exception:
                    total_gpu_mem = 0
                assert (
                    config[GPU_MEMORY_LIMIT] <= total_gpu_mem
                ), "Requested GPU memory limit is greater than total GPU memory"
                conf_dict[GPU_MEMORY_LIMIT] = config[GPU_MEMORY_LIMIT]

    return conf_dict


def get_devices(gpu_config=None):
    """Devices (``torch.device`` list) of the ranks of a fit; the first one is the main process.

    Precedence: ``TORCHPOTENTIAL_DEVICES`` (comma-separated, e.g. ``cuda:0,cuda:1``), ``TORCHPOTENTIAL_DEVICE``,
    ``gpu_config['gpu_ind']`` (int or list), else ``cuda:0`` / CPU. More than one device means a multi-process fit.
    """
    import os

    env_list = os.environ.get(_device.ENV_DEVICES)
    if env_list:
        devs = [torch.device(d.strip()) for d in env_list.split(",") if d.strip()]
        if devs:
            return devs
    if os.environ.get(_device.ENV_DEVICE):
        return [torch.device(os.environ[_device.ENV_DEVICE])]

    idx = _set_gpu_config(gpu_config)[GPU_INDEX]
    indices = idx if isinstance(idx, list) else [idx]
    if not torch.cuda.is_available():
        if isinstance(idx, list):
            raise RuntimeError("{} {} requested, but CUDA is not available".format(GPU_INDEX, idx))
        return [torch.device("cpu")]
    if not isinstance(idx, list) and idx < 0:
        return [torch.device("cpu")]
    n = torch.cuda.device_count()
    for i in indices:
        assert i < n, "GPU ind {} is requested, but there are only {} GPUs".format(i, n)
    return [torch.device("cuda", i) for i in indices]


def apply_memory_limit(device, mem_limit):
    """Cap the caching allocator of this process on ``device`` to ``mem_limit`` MB (0 = no limit)."""
    device = torch.device(device)
    if mem_limit and device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        frac = mem_limit / get_gpu_memory(idx)
        torch.cuda.set_per_process_memory_fraction(min(frac, 1.0), idx)


def init_gpu_config(gpu_config):
    """Select the compute device of this process. ``gpu_ind < 0`` -> CPU; ``mem_limit`` (MB) caps the caching allocator.

    With a list in ``gpu_ind`` (or ``TORCHPOTENTIAL_DEVICES``) the first entry is the device of this (main) process;
    use :func:`get_devices` for the full list. ``TORCHPOTENTIAL_DEVICE``, if set, takes precedence over the config.
    """
    import os

    gpu_config = _set_gpu_config(gpu_config)
    devices = get_devices(gpu_config)
    _device.set_device(devices[0])
    if devices[0].type == "cuda" and not os.environ.get(_device.ENV_DEVICE) and not os.environ.get(_device.ENV_DEVICES):
        torch.cuda.set_device(devices[0])
        apply_memory_limit(devices[0], gpu_config[GPU_MEMORY_LIMIT])

    return gpu_config
