"""Multi-device fits: batches spread over worker processes must reproduce the single-process fit."""
import os
import signal

import numpy as np
import pytest
import torch

from torchpotential.distributed import DevicePool, WorkerError, merge_results, shard_batches
from torchpotential.utils import device as device_utils
from torchpotential.utils.utilities import get_devices, _set_gpu_config

from fit_cases import CASES
from test_fit_vs_tf import run_case

CASE = {c["name"]: c for c in CASES}


def _case(name, **kw):
    c = dict(CASE[name])
    c.update(kw)
    return c


@pytest.fixture(scope="module")
def pool2():
    p = DevicePool(["cpu", "cpu"], num_threads=1)
    yield p
    p.close()


@pytest.fixture(scope="module")
def pool3():
    p = DevicePool(["cpu", "cpu", "cpu"], num_threads=1)
    yield p
    p.close()


def _evaluate(fitter, df, batch_size, x):
    """Loss, gradient and errors of one evaluation at ``x`` (a fitter that has set up its batches)."""
    batches = fitter._train_batches
    tot_loss, tot_jac, errors = fitter._evaluate_all(x, batches)
    return tot_loss, tot_jac, errors


# ------------------------------------------------------------------------------------------------
# sharding
# ------------------------------------------------------------------------------------------------
def test_shard_batches_covers_every_batch_once_and_is_balanced():
    rng = np.random.default_rng(0)
    costs = list(rng.integers(50, 500, size=23))
    shards = shard_batches(costs, 4)
    assert sorted(i for s in shards for i in s) == list(range(23))
    assert all(len(s) >= 1 for s in shards)
    assert all(s == sorted(s) for s in shards)
    loads = [sum(costs[i] for i in s) for s in shards]
    assert max(loads) - min(loads) <= max(costs)
    assert shards == shard_batches(costs, 4)
    assert int(np.argmax(costs)) in shards[0]


def test_shard_batches_with_fewer_batches_than_ranks():
    shards = shard_batches([5, 9], 4)
    assert [len(s) for s in shards] == [1, 1, 0, 0]
    assert shards[0] == [1]


# ------------------------------------------------------------------------------------------------
# device selection
# ------------------------------------------------------------------------------------------------
def test_gpu_config_accepts_lists_and_rejects_bad_values():
    assert _set_gpu_config({"gpu_ind": [0, 0, 1]})["gpu_ind"] == [0, 0, 1]
    assert _set_gpu_config({"gpu_ind": 2})["gpu_ind"] == 2
    for bad in ([], [0, -1], [0, "1"], [True]):
        with pytest.raises(AssertionError):
            _set_gpu_config({"gpu_ind": bad})


def test_get_devices_env_precedence(monkeypatch):
    monkeypatch.setenv(device_utils.ENV_DEVICES, "cpu, cpu,cpu")
    assert [str(d) for d in get_devices({"gpu_ind": 0})] == ["cpu", "cpu", "cpu"]
    monkeypatch.delenv(device_utils.ENV_DEVICES)
    monkeypatch.setenv(device_utils.ENV_DEVICE, "cpu")
    assert [str(d) for d in get_devices({"gpu_ind": [0, 1]})] == ["cpu"]
    monkeypatch.delenv(device_utils.ENV_DEVICE)
    assert [str(d) for d in get_devices({"gpu_ind": -1})] == ["cpu"]
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            get_devices({"gpu_ind": [0, 1]})


# ------------------------------------------------------------------------------------------------
# parity with the single-process evaluation
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["bfgs_unary_aux", "lbfgsb_binary_mask"])
@pytest.mark.parametrize("pool_name", ["pool2", "pool3"])
def test_distributed_fit_matches_serial(name, pool_name, request):
    pool = request.getfixturevalue(pool_name)
    case = _case(name, batch_size=1 if name == "bfgs_unary_aux" else 2)  # 5 batches / 3 batches
    serial, final_s, df = run_case(case)
    dist, final_d, _ = run_case(case, start=None, pool=pool)
    np.testing.assert_allclose(dist.loss_history, serial.loss_history, rtol=1e-9)
    np.testing.assert_allclose(final_d, final_s, rtol=1e-7, atol=1e-9)
    # metrics are assembled from the merged per-batch errors in batch order
    for key in ("rmse_epa", "rmse_f", "mae_f", "max_abs_f", "e_loss", "f_loss"):
        np.testing.assert_allclose(dist.last_fit_metric_data[key], serial.last_fit_metric_data[key], rtol=1e-8)
    np.testing.assert_allclose(
        dist.last_fit_metric_data["extra_regularization_contrib"],
        serial.last_fit_metric_data["extra_regularization_contrib"],
        rtol=1e-9,
    )
    assert dist._local_idx is not None


@pytest.mark.parametrize("pool_name", ["pool2", "pool3"])
def test_distributed_loss_and_gradient_match_serial(pool_name, request):
    pool = request.getfixturevalue(pool_name)
    case = _case("bfgs_unary_aux", batch_size=1, niter=1)
    serial, _, _ = run_case(case)
    dist, _, _ = run_case(case, pool=pool)
    rng = np.random.default_rng(5)
    x = serial.fit_coefs + 0.01 * rng.normal(size=serial.fit_coefs.shape)
    ls, gs, es = serial._evaluate_all(x, serial._train_batches)
    ld, gd, ed = dist._evaluate_all(x, dist._train_batches)
    np.testing.assert_allclose(ld, ls, rtol=1e-12)
    np.testing.assert_allclose(gd, gs, rtol=1e-10, atol=1e-13 * np.abs(gs).max())
    assert len(ed) == len(es)
    for a, b in zip(ed, es):
        for u, v in zip(a, b):
            np.testing.assert_allclose(u, v, rtol=1e-10, atol=1e-12)


def test_idle_ranks_when_there_are_fewer_batches_than_ranks(pool3):
    case = _case("lbfgsb_binary_mask", batch_size=3, niter=3)  # 2 batches, 3 ranks
    serial, final_s, _ = run_case(case)
    dist, final_d, _ = run_case(case, pool=pool3)
    assert pool3.active == [1]
    np.testing.assert_allclose(dist.loss_history, serial.loss_history, rtol=1e-9)
    np.testing.assert_allclose(final_d, final_s, rtol=1e-7, atol=1e-9)


def test_pool_is_reused_by_a_second_fit(pool2):
    case = _case("bfgs_unary_aux", batch_size=1, niter=3)
    first, _, _ = run_case(case, pool=pool2)
    second, final, _ = run_case(case, pool=pool2)
    np.testing.assert_allclose(second.loss_history, first.loss_history, rtol=1e-12)


def test_predict_after_distributed_fit(pool2):
    case = _case("bfgs_unary_aux", batch_size=1, niter=3)
    serial, _, df = run_case(case)
    dist, _, _ = run_case(case, pool=pool2)
    ps, pd_ = serial.predict(df.copy()), dist.predict(df.copy())
    np.testing.assert_allclose(pd_["energy_pred"], ps["energy_pred"], rtol=1e-8, atol=1e-9)
    # only this process's share of the training batches stays on the device
    assert len(dist._tbatches) == len(dist._local_idx)


# ------------------------------------------------------------------------------------------------
# unsupported optimizers
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("optimizer", ["Adam", "SGD", "trf", "dogbox", "lm"])
def test_unsupported_optimizers_are_rejected(optimizer, pool2):
    case = _case("bfgs_unary_aux", batch_size=1, niter=1, optimizer=optimizer)
    with pytest.raises(NotImplementedError):
        run_case(case, pool=pool2)


# ------------------------------------------------------------------------------------------------
# failures
# ------------------------------------------------------------------------------------------------
def test_worker_failure_reaches_the_main_process_and_the_pool_recovers(pool2):
    case = _case("bfgs_unary_aux", batch_size=1, niter=1)
    fitter, _, _ = run_case(case, pool=pool2)
    good = fitter.fit_coefs
    with pytest.raises(WorkerError, match="worker rank 1 failed during evaluation"):
        pool2.evaluate(np.zeros(3), lambda: (0.0, 0.0, []))  # wrong number of coefficients on the worker
    # the failed worker dropped its state; a new setup brings it back
    pool2.setup(fitter.tensorpot.potential, fitter.tensorpot.loss_specs,
                [[(i, fitter._train_batches[i]) for i in range(len(fitter._train_batches)) if i not in fitter._local_idx]])
    loss, grad, errors = fitter._evaluate_all(good, fitter._train_batches)
    assert np.isfinite(loss) and np.all(np.isfinite(grad)) and len(errors) == len(fitter._train_batches)


def test_main_process_failure_does_not_desynchronise_the_pool(pool2):
    case = _case("bfgs_unary_aux", batch_size=1, niter=1)
    fitter, _, _ = run_case(case, pool=pool2)
    good = fitter.fit_coefs

    def boom():
        raise ValueError("local failure")

    with pytest.raises(ValueError, match="local failure"):
        pool2.evaluate(good, boom)
    loss, grad, errors = fitter._evaluate_all(good, fitter._train_batches)
    assert np.isfinite(loss) and len(errors) == len(fitter._train_batches)


def test_oom_text_of_a_worker_is_recognised():
    err = WorkerError("worker rank 1 failed during evaluation: OutOfMemoryError: CUDA out of memory. Tried to allocate")
    assert device_utils.is_oom_error(err)


def test_dead_worker_is_reported_instead_of_hanging():
    pool = DevicePool(["cpu", "cpu"], num_threads=1, timeout=60)
    try:
        case = _case("bfgs_unary_aux", batch_size=1, niter=1)
        fitter, _, _ = run_case(case, pool=pool)
        os.kill(pool._procs[0].pid, signal.SIGKILL)
        pool._procs[0].wait()
        with pytest.raises(WorkerError, match="rank 1"):
            fitter._evaluate_all(fitter.fit_coefs, fitter._train_batches)
        with pytest.raises(WorkerError):  # a broken pool refuses further work
            fitter._evaluate_all(fitter.fit_coefs, fitter._train_batches)
    finally:
        pool.close()


def test_merge_results_orders_errors_by_batch_index():
    a = (1.0, np.ones(2), [(2, "c"), (0, "a")])
    b = (2.0, 2 * np.ones(2), [(1, "b")])
    loss, grad, errors = merge_results([a, b])
    assert loss == 3.0 and np.all(grad == 3.0) and errors == ["a", "b", "c"]


# ------------------------------------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_two_processes_on_one_gpu_match_serial():
    pool = DevicePool(["cuda:0", "cuda:0"], num_threads=1)
    try:
        case = _case("bfgs_unary_aux", batch_size=1, niter=4)
        serial, final_s, _ = run_case(case, device="cpu")
        dist, final_d, _ = run_case(case, pool=pool, device="cuda:0")
        np.testing.assert_allclose(dist.loss_history, serial.loss_history, rtol=1e-8)
        np.testing.assert_allclose(final_d, final_s, rtol=1e-6, atol=1e-8)
    finally:
        pool.close()
