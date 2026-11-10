import logging
import os

import torch

log = logging.getLogger(__name__)

ENV_DEVICE = "TORCHPOTENTIAL_DEVICE"
ENV_NUM_THREADS = "TORCHPOTENTIAL_NUM_THREADS"
_DEVICE = None


def _is_xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_device():
    """Device used for new potentials/batches: env override, else the one set by init_gpu_config, else cuda:0/xpu:0/CPU."""
    global _DEVICE
    if _DEVICE is None:
        env = os.environ.get(ENV_DEVICE)
        if env:
            _DEVICE = torch.device(env)
        elif torch.cuda.is_available():
            _DEVICE = torch.device("cuda", 0)
        elif _is_xpu_available():
            _DEVICE = torch.device("xpu", 0)
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE


def set_device(device):
    global _DEVICE
    _DEVICE = torch.device(device)
    return _DEVICE


def is_oom_error(exc):
    """True if ``exc`` is a (CUDA/XPU) out-of-memory error."""
    oom_types = []
    if hasattr(torch, "OutOfMemoryError"):
        oom_types.append(torch.OutOfMemoryError)
    if hasattr(torch.cuda, "OutOfMemoryError"):
        oom_types.append(torch.cuda.OutOfMemoryError)
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "OutOfMemoryError"):
        oom_types.append(torch.xpu.OutOfMemoryError)
    for oom in oom_types:
        try:
            if isinstance(exc, oom):
                return True
        except Exception:
            pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


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
