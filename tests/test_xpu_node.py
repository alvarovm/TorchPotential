"""Opt-in full-node test: real fitting and audited tensor placement on 12 distinct XPUs."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from pyace.basis import BBasisConfiguration
from torchpotential.constants import LOSS_TYPE, LOSS_FORCE_FACTOR, LOSS_ENERGY_FACTOR, L1_REG, L2_REG
from torchpotential.distributed import DevicePool, evaluate_batches, merge_results
from torchpotential.fit import FitTorchPotential
from torchpotential.potentials.ace import ACE
from torchpotential.torchpot import TorchPotential
from torchpotential.utils import device as device_utils
from torchpotential.utils.utilities import get_devices, init_gpu_config

from helpers import make_df, make_structures
from xpu_audit_worker import audited_external_fit


@pytest.mark.skipif(os.environ.get("TORCHPOTENTIAL_RUN_12_XPU_TEST") != "1",
                    reason="requires an allocation of 12 XPUs; set TORCHPOTENTIAL_RUN_12_XPU_TEST=1")
def test_twelve_xpus_fit_and_evaluate_every_shard(tmp_path, monkeypatch):
    # Once explicitly requested, missing hardware must fail rather than silently skip.
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "XPU-enabled PyTorch and an allocated node are required"
    assert torch.xpu.device_count() >= 12, "This test requires 12 visible, allocated XPU devices"
    expected_devices = ["xpu:{}".format(i) for i in range(12)]
    monkeypatch.setenv(device_utils.ENV_DEVICES, ",".join(expected_devices))
    monkeypatch.setenv(device_utils.ENV_NUM_THREADS, "1")
    monkeypatch.setenv("TORCHPOTENTIAL_XPU_AUDIT_DIR", str(tmp_path))
    monkeypatch.setattr(device_utils, "_DEVICE", None)
    device_utils.set_num_threads(1)
    config = init_gpu_config({"gpu_ind": list(range(12)), "mem_limit": 8192})
    devices = get_devices(config)
    assert [str(d) for d in devices] == expected_devices

    # Equal-cost synthetic batches give exactly four nonempty batches to every rank.
    basis = BBasisConfiguration(str(Path(__file__).parent / "data/yaml/Al-r1234l12_crad_dif_2.yaml"))
    for block in basis.funcspecs_blocks:
        block.core_rep_parameters = [0.0, 1.0]
    cutoff = max(b.rcutij for b in basis.funcspecs_blocks) + 0.05
    structure = make_structures(["Al"], seed=21, min_dist=2.0)[0]
    df = make_df([structure] * 48, cutoff, seed=21)
    specs = {LOSS_TYPE: "per-atom", LOSS_FORCE_FACTOR: 0.4, LOSS_ENERGY_FACTOR: 0.6,
             L1_REG: 1e-5 / 48, L2_REG: 1e-5 / 48}
    cpu = FitTorchPotential(TorchPotential(ACE(basis, device="cpu"), loss_specs=specs), display_step=1000)
    start = cpu.tensorpot.potential.get_coefs().detach().cpu().numpy().copy()
    start += 0.01 * np.random.default_rng(42).normal(size=start.shape)
    cpu.fit(df.copy(), batch_size=1, niter=2, optimizer="BFGS", coefs=start, fit_metric_callback=lambda d: None)

    # Only the entry point is instrumented: the actual DevicePool protocol, worker loop and XPU kernels are unchanged.
    worker_script = str(Path(__file__).with_name("xpu_audit_worker.py"))
    popen = subprocess.Popen

    def audited_worker(command, *args, **kwargs):
        command = list(command)
        if command[1:3] == ["-m", "torchpotential.distributed"]:
            command[1:3] = [worker_script]
        return popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", audited_worker)
    external_fit = TorchPotential.external_fit
    monkeypatch.setattr(TorchPotential, "external_fit", audited_external_fit(devices[0], tmp_path / "rank_0.json"))
    pool = DevicePool(devices, num_threads=1, mem_limit=config["mem_limit"], timeout=180)
    procs = list(pool._procs)
    try:
        gpu = FitTorchPotential(TorchPotential(ACE(basis, device=devices[0]), loss_specs=specs),
                                pool=pool, display_step=1000)
        gpu.fit(df.copy(), batch_size=1, niter=2, optimizer="BFGS", coefs=start,
                fit_metric_callback=lambda d: None)
        assert pool.world_size == 12 and len(procs) == 11
        assert pool.active == list(range(1, 12)), "Every worker must hold training batches"
        assert len(gpu._local_idx) == 4
        np.testing.assert_allclose(gpu.loss_history, cpu.loss_history, rtol=1e-8)
        np.testing.assert_allclose(gpu.fit_coefs, cpu.fit_coefs, rtol=1e-6, atol=1e-8)

        x = gpu.fit_coefs.copy()
        local = [(i, gpu._train_batches[i], gpu._tbatch(gpu._train_batches[i])) for i in gpu._local_idx]
        results = pool.evaluate(x, lambda: evaluate_batches(gpu.tensorpot, x, local))
        assert len(results) == 12
        assert [len(errors) for _, _, errors in results] == [4] * 12
        indices = [i for _, _, errors in results for i, _ in errors]
        assert sorted(indices) == list(range(48)), "Every batch must be evaluated exactly once"
        for loss, grad, _ in results:
            assert np.isfinite(loss) and loss > 0
            assert np.all(np.isfinite(grad)) and np.linalg.norm(grad) > 0
        loss, grad, errors = merge_results(results)
        # Temporarily bypass the XPU audit for the CPU reference.
        with monkeypatch.context() as m:
            m.setattr(TorchPotential, "external_fit", external_fit)
            cpu_loss, cpu_grad, cpu_errors = cpu._evaluate_all(x, cpu._train_batches)
        np.testing.assert_allclose(loss, cpu_loss, rtol=1e-10)
        np.testing.assert_allclose(grad, cpu_grad, rtol=1e-9, atol=1e-11)
        assert len(errors) == len(cpu_errors) == 48
        for actual, expected in zip(errors, cpu_errors):
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e, rtol=1e-9, atol=1e-10)

        print("\nrank  pid      device  batches  allocated MiB  loss/gradient/energy/forces device")
        pids = [os.getpid()] + [p.pid for p in procs]
        for rank, expected in enumerate(expected_devices):
            report = json.loads((tmp_path / "rank_{}.json".format(rank)).read_text())
            assert report["pid"] == pids[rank]
            assert report["device"] == report["model_device"] == report["batch_device"] == expected
            assert report["result_devices"] == [expected] * 4
            assert report["allocated_bytes"] > 0
            print("{:>4}  {:<7}  {:<6}  {:>7}  {:>13.2f}  {}".format(
                rank, report["pid"], expected, len(results[rank][2]),
                report["allocated_bytes"] / 1024**2, ", ".join(report["result_devices"])))
        assert len(set(pids)) == 12
        print("All 12 XPUs evaluated their shards; CPU loss and gradients match.")
        pool.release()
        assert pool.active == []
    finally:
        pool.close()
    assert all(p.poll() is not None for p in procs)
