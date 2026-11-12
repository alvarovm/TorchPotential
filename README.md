# TorchPotential

PyTorch port of the part of TensorPotential (`tensorpotential.compat.pace`) that
`python-ace` (`pyace` / `pacemaker`) uses to fit Atomic Cluster Expansion potentials.
It replaces the TensorFlow implementation; TensorFlow is not a dependency.

Public classes (API-compatible with the TF versions):

| module | class |
|---|---|
| `torchpotential.potentials.ace` | `ACE` |
| `torchpotential.torchpot` | `TorchPotential` |
| `torchpotential.fit` | `FitTorchPotential` |

Everything runs in float64. The device defaults to `cuda:0` when available, then `xpu:0`, otherwise CPU;
override with `init_gpu_config({"gpu_ind": -1})` (CPU), `{"gpu_ind": N, "mem_limit": MB}`,
or the `TORCHPOTENTIAL_DEVICE` environment variable (e.g. `cpu`, `cuda:1`, `xpu:1`). Intel XPUs must support FP64.

## Several devices

The fit loss is a sum over batches, so the batches can be spread over several devices. Set `gpu_ind` to a list
(`init_gpu_config({"gpu_ind": [0, 1, 2, 3]})`) or use `TORCHPOTENTIAL_DEVICES=cuda:0,cuda:1` (NVIDIA) /
`TORCHPOTENTIAL_DEVICES=xpu:0,xpu:1` (Intel), create a
`torchpotential.distributed.DevicePool` for these devices and pass it to `FitTorchPotential(..., pool=pool)`:

```python
from torchpotential.distributed import DevicePool
from torchpotential.fit import FitTorchPotential
from torchpotential.potentials.ace import ACE
from torchpotential.torchpot import TorchPotential
from torchpotential.utils.device import set_num_threads
from torchpotential.utils.utilities import get_devices, init_gpu_config

# bbasis: pyace BBasisConfiguration; df: dataframe with the prepared tp_atoms column
config = init_gpu_config({"gpu_ind": [0, 1], "mem_limit": 0})
devices = get_devices(config)                 # CUDA if available, otherwise XPU; env overrides are honored
num_threads = set_num_threads(2)              # honors TORCHPOTENTIAL_NUM_THREADS
pool = DevicePool(devices, num_threads=num_threads, mem_limit=config["mem_limit"]) if len(devices) > 1 else None
try:
    fitter = FitTorchPotential(TorchPotential(ACE(bbasis, device=devices[0])), pool=pool)
    fitter.fit(df, optimizer="BFGS", batch_size=1000)
finally:
    if pool is not None:
        pool.close()
```

`init_gpu_config` initializes rank 0 and its allocator limit; `DevicePool` initializes the workers with the same
limit. The single-device fallback handles an environment override such as `TORCHPOTENTIAL_DEVICE=cpu` without
creating a pool. `finally` ensures workers stop even when fitting raises an exception.

The main process runs the optimizer and evaluates its own share of the batches; each worker process holds a copy of the
potential and its share of the batches. Per evaluation only the coefficients, the summed gradient and the residuals are
exchanged. The batches are the same as in a single-device fit, so loss, regularisation and metrics are identical
(up to summation order). Repeating an index (`[0, 0]`) runs several processes on one GPU. Supported optimizers:
`BFGS`, `L-BFGS-B`, `CG`, `dual_annealing`. Single-node, POSIX only; no MPI or DDP launcher is needed.
`TORCHPOTENTIAL_DEVICES` takes precedence over `TORCHPOTENTIAL_DEVICE`, then `gpu_ind`. `mem_limit` (MB) caps
each process's CUDA/XPU caching allocator, including rank 0 when devices are selected through the environment;
use `init_gpu_config(config)` in rank 0 and pass the same limit to `DevicePool(..., mem_limit=...)`.
A fit is only faster when each device has enough work: use at least 4 batches per device. The observed CUDA
launch-bound threshold (~30000 atoms per batch) is hardware-dependent, not a guarantee of XPU scaling.
`TORCHPOTENTIAL_WORKER_TIMEOUT` (seconds) bounds the wait for a worker; a dead worker is always detected.

With `pacemaker`, no Python pool setup is necessary. Merge this into an existing input file:

```yaml
fit:
  optimizer: BFGS
backend:
  evaluator: torchpot
  batch_size: 10
  num_threads: 4
  gpu_config: {gpu_ind: [0, 1], mem_limit: 0}
```

Unset device environment overrides to use the YAML indices, or run
`TORCHPOTENTIAL_DEVICES=xpu:0,xpu:1 pacemaker input.yaml` to explicitly select Intel GPUs (`cuda:0,cuda:1` for NVIDIA).
Install compatible `distributed` versions of both repositories for this implementation, and launch one main process,
not one rank per GPU. The test set stays on the first device. The python-ace documentation has the full
[multi-GPU tutorial](https://github.com/alvarovm/python-ace/blob/distributed/docs/pacemaker/multi_gpu.md) and
[Aurora/XPU setup](https://github.com/alvarovm/python-ace/blob/distributed/docs/pacemaker/runxpu.md).

Core repulsion, embedding and inner cutoff follow the C++ ACE evaluator (`inner_cutoff_type` `density` and
`distance`; `zbl` is not implemented and raises `NotImplementedError`).

## Install and test

```bash
pip install torch            # CPU, CUDA or XPU build, see pytorch.org (or use the site's frameworks module)
pip install .                # needs `pyace` (python-ace) at runtime
OMP_NUM_THREADS=2 python -m pytest tests -q
```

To verify all 12 XPUs of an allocated Aurora node, activate the site's environment and run from this checkout:

```bash
TORCHPOTENTIAL_RUN_12_XPU_TEST=1 TORCHPOTENTIAL_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_xpu_node.py -v -s
```

This opt-in test fits 48 synthetic batches, checks four real batches per device, audits tensor placement and
allocator usage in all 12 processes, and compares loss/gradients and fit coefficients with CPU. It prints per-rank
device evidence and closes all workers. Without the opt-in variable it skips; when enabled, missing hardware fails.
Only enable it when all 12 devices are allocated to your job. It is not a scaling benchmark.

On busy machines limit the CPU threads (`TORCHPOTENTIAL_NUM_THREADS` or `torch.set_num_threads`): the default
(all cores) can be orders of magnitude slower than a single thread for the small tensors of ACE fits.

Derived from grace-tensorpotential; distributed under the same Academic Software Licence (see `LICENSE.md`).
