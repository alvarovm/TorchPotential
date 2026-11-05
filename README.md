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

Everything runs in float64. The device defaults to `cuda:0` when available, otherwise CPU;
override with `init_gpu_config({"gpu_ind": -1})` (CPU), `{"gpu_ind": N, "mem_limit": MB}`,
or the `TORCHPOTENTIAL_DEVICE` environment variable (e.g. `cpu`, `cuda:1`).

## Several devices

The fit loss is a sum over batches, so the batches can be spread over several devices. Set `gpu_ind` to a list
(`init_gpu_config({"gpu_ind": [0, 1, 2, 3]})`) or use `TORCHPOTENTIAL_DEVICES=cuda:0,cuda:1`, create a
`torchpotential.distributed.DevicePool` for these devices and pass it to `FitTorchPotential(..., pool=pool)`:

```python
from torchpotential.distributed import DevicePool
from torchpotential.utils.utilities import get_devices

devices = get_devices({"gpu_ind": [0, 1]})
pool = DevicePool(devices, num_threads=2)       # starts one worker process per device after the first
fitter = FitTorchPotential(TorchPotential(ACE(bbasis, device=devices[0])), pool=pool)
fitter.fit(df, optimizer="BFGS", batch_size=1000)
pool.close()
```

The main process runs the optimizer and evaluates its own share of the batches; each worker process holds a copy of the
potential and its share of the batches. Per evaluation only the coefficients, the summed gradient and the residuals are
exchanged. The batches are the same as in a single-device fit, so loss, regularisation and metrics are identical
(up to summation order). Repeating an index (`[0, 0]`) runs several processes on one GPU. Supported optimizers:
`BFGS`, `L-BFGS-B`, `CG`, `dual_annealing`. POSIX only. A fit is only faster when each device has enough work: an
evaluation of one batch is launch-bound below ~30000 atoms, and use at least 4 batches per device.
`TORCHPOTENTIAL_WORKER_TIMEOUT` (seconds) bounds the wait for a worker; a dead worker is always detected.

Core repulsion, embedding and inner cutoff follow the C++ ACE evaluator (`inner_cutoff_type` `density` and
`distance`; `zbl` is not implemented and raises `NotImplementedError`).

## Install and test

```bash
pip install torch            # CPU or CUDA build, see pytorch.org
pip install .                # needs `pyace` (python-ace) at runtime
OMP_NUM_THREADS=2 python -m pytest tests -q
```

On busy machines limit the CPU threads (`TORCHPOTENTIAL_NUM_THREADS` or `torch.set_num_threads`): the default
(all cores) can be orders of magnitude slower than a single thread for the small tensors of ACE fits.

Derived from grace-tensorpotential; distributed under the same Academic Software Licence (see `LICENSE.md`).
