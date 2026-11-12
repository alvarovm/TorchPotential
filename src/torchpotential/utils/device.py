import logging
import os

import torch

log = logging.getLogger(__name__)

ENV_DEVICE = "TORCHPOTENTIAL_DEVICE"
ENV_DEVICES = "TORCHPOTENTIAL_DEVICES"
ENV_NUM_THREADS = "TORCHPOTENTIAL_NUM_THREADS"
_DEVICE = None


def _is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _available_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if _is_xpu_available():
        return "xpu"
    return None


def get_device():
    """Device for new potentials/batches: configured device, env override, else cuda:0/xpu:0/CPU."""
    global _DEVICE
    if _DEVICE is None:
        env_devices = [d.strip() for d in os.environ.get(ENV_DEVICES, "").split(",") if d.strip()]
        env = env_devices[0] if env_devices else os.environ.get(ENV_DEVICE)
        accelerator = _available_accelerator() if not env else None
        if env:
            _DEVICE = torch.device(env)
        elif accelerator:
            _DEVICE = torch.device(accelerator, 0)
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE


def set_device(device):
    global _DEVICE
    _DEVICE = torch.device(device)
    return _DEVICE


def is_oom_error(exc):
    """True for CUDA/XPU out-of-memory errors, including errors forwarded by a worker.

    Covers the typed ``OutOfMemoryError`` exceptions, the common CUDA/XPU "out of memory" message, and the
    Level-Zero/SYCL resource-exhaustion errors Intel XPUs raise instead (e.g. ``UR_RESULT_ERROR_OUT_OF_RESOURCES``,
    ``ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY``/``..._OUT_OF_HOST_MEMORY``): confirmed on real hardware to fire when a
    device is simply out of memory headroom (e.g. a large, permanently resident test set/shard leaving no room for
    one more batch), with no "out of memory" substring in the message, so it would otherwise bypass
    ``batch_size_reduction`` and crash the whole fit instead of retrying with a smaller batch.
    """
    oom_types = tuple(
        oom for module in (torch, torch.cuda, getattr(torch, "xpu", None))
        if isinstance(oom := getattr(module, "OutOfMemoryError", None), type)
    )
    if isinstance(exc, oom_types):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    if "out of memory" in message:
        return True
    resource_exhaustion = (
        "ur_result_error_out_of_resources",
        "ze_result_error_out_of_device_memory",
        "ze_result_error_out_of_host_memory",
    )
    return any(pattern in message for pattern in resource_exhaustion)


def set_num_threads(num_threads=None):
    """Set the number of CPU threads used by torch.

    ``TORCHPOTENTIAL_NUM_THREADS`` takes precedence over ``num_threads``. ``None``/0 keeps torch's default.
    The default (all cores) can be very slow on a busy/oversubscribed machine, where a few threads
    or a single one is much faster for the small tensors used in ACE fits.
    """
    env = os.environ.get(ENV_NUM_THREADS)
    n = int(env) if env else num_threads
    if n:
        torch.set_num_threads(int(n))
        log.info("torch CPU threads: %d", int(n))
    return torch.get_num_threads()
