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
