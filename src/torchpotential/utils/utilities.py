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


def _available_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return None


def _device_count(accel):
    if accel == "cuda":
        return torch.cuda.device_count()
    if accel == "xpu":
        return torch.xpu.device_count()
    return 0


def _set_device_index(accel, idx):
    if accel == "cuda":
        torch.cuda.set_device(idx)
    elif accel == "xpu":
        torch.xpu.set_device(idx)


def _set_memory_fraction(accel, frac, idx):
    if accel == "cuda":
        torch.cuda.set_per_process_memory_fraction(frac, idx)
    elif accel == "xpu":
        torch.xpu.set_per_process_memory_fraction(frac, idx)


def _get_device_properties(accel, idx):
    if accel == "cuda":
        return torch.cuda.get_device_properties(idx)
    if accel == "xpu":
        return torch.xpu.get_device_properties(idx)
    raise RuntimeError("No accelerator available")


def get_gpu_memory(gpu_id):
    """Total memory of the GPU in MB."""
    accel = _available_accelerator()
    if accel is None:
        return 0
    try:
        return _get_device_properties(accel, gpu_id).total_memory // (1024 * 1024)
    except Exception:
        return 0


def _set_gpu_config(config=None):
    conf_dict = {GPU_INDEX: 0, GPU_MEMORY_LIMIT: 0}

    if config is not None:
        assert isinstance(config, dict), "gpu_config must be a dict"
        if GPU_INDEX in config:
            assert isinstance(config[GPU_INDEX], int), "{} must be an integer".format(GPU_INDEX)
            conf_dict[GPU_INDEX] = config[GPU_INDEX]

        if GPU_MEMORY_LIMIT in config:
            assert isinstance(config[GPU_MEMORY_LIMIT], int), "{} must be an integer number of MB".format(
                GPU_MEMORY_LIMIT
            )
            if config[GPU_MEMORY_LIMIT] < 0:
                raise ValueError("{} must be not negative".format(GPU_MEMORY_LIMIT))
            if conf_dict[GPU_INDEX] >= 0 and config[GPU_MEMORY_LIMIT] > 0:
                try:
                    total_gpu_mem = get_gpu_memory(conf_dict[GPU_INDEX])
                except Exception:
                    total_gpu_mem = 0
                assert (
                    config[GPU_MEMORY_LIMIT] <= total_gpu_mem
                ), "Requested GPU memory limit is greater than total GPU memory"
                conf_dict[GPU_MEMORY_LIMIT] = config[GPU_MEMORY_LIMIT]

    return conf_dict


def init_gpu_config(gpu_config):
    """Select the compute device. ``gpu_ind < 0`` -> CPU; ``mem_limit`` (MB) caps the caching allocator.

    The ``TORCHPOTENTIAL_DEVICE`` environment variable, if set, takes precedence.
    """
    import os

    gpu_config = _set_gpu_config(gpu_config)
    if os.environ.get(_device.ENV_DEVICE):
        _device.set_device(os.environ[_device.ENV_DEVICE])
        return gpu_config

    idx = gpu_config[GPU_INDEX]
    accel = _available_accelerator()
    if idx < 0 or accel is None:
        _device.set_device("cpu")
    else:
        n = _device_count(accel)
        assert idx < n, "GPU ind {} is requested, but there are only {} {} GPUs".format(idx, n, accel)
        _device.set_device(torch.device(accel, idx))
        _set_device_index(accel, idx)
        if gpu_config[GPU_MEMORY_LIMIT] != 0:
            frac = gpu_config[GPU_MEMORY_LIMIT] / get_gpu_memory(idx)
            _set_memory_fraction(accel, min(frac, 1.0), idx)

    return gpu_config
