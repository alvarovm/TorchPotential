"""CUDA/XPU selection and allocator configuration without requiring GPU hardware."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchpotential import distributed
from torchpotential.utils import device as device_utils
from torchpotential.utils.utilities import apply_memory_limit, get_devices, get_gpu_memory, init_gpu_config


@pytest.fixture
def accelerators(monkeypatch):
    monkeypatch.delenv(device_utils.ENV_DEVICE, raising=False)
    monkeypatch.delenv(device_utils.ENV_DEVICES, raising=False)
    monkeypatch.setattr(device_utils, "_DEVICE", None)
    apis = {}
    for kind in ("cuda", "xpu"):
        api = SimpleNamespace(
            is_available=Mock(return_value=False), device_count=Mock(return_value=2),
            set_device=Mock(), current_device=Mock(return_value=1), empty_cache=Mock(),
            get_device_properties=Mock(return_value=SimpleNamespace(total_memory=8192 * 1024 * 1024)),
            set_per_process_memory_fraction=Mock(), OutOfMemoryError=type(kind + "OOM", (RuntimeError,), {}),
        )
        monkeypatch.setattr(torch, kind, api, raising=False)
        apis[kind] = api
    return apis


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_auto_and_list_selection(accelerators, kind):
    accelerators[kind].is_available.return_value = True
    assert str(device_utils.get_device()) == kind + ":0"
    assert [str(d) for d in get_devices({"gpu_ind": [1, 0, 1]})] == [kind + ":1", kind + ":0", kind + ":1"]
    assert [str(d) for d in get_devices({"gpu_ind": -1})] == ["cpu"]
    with pytest.raises(AssertionError, match="only 2"):
        get_devices({"gpu_ind": [0, 2]})


def test_cuda_is_preferred_when_both_are_available(accelerators):
    for api in accelerators.values():
        api.is_available.return_value = True
    assert str(device_utils.get_device()) == "cuda:0"
    assert [str(d) for d in get_devices({"gpu_ind": [0, 1]})] == ["cuda:0", "cuda:1"]


def test_no_accelerator_falls_back_to_cpu_but_rejects_gpu_lists(accelerators):
    assert str(device_utils.get_device()) == "cpu"
    assert [str(d) for d in get_devices()] == ["cpu"]
    assert get_gpu_memory(0) == 0
    with pytest.raises(RuntimeError, match="neither CUDA nor XPU"):
        get_devices({"gpu_ind": [0, 1]})


def test_pytorch_without_xpu_attribute(accelerators, monkeypatch):
    monkeypatch.delattr(torch, "xpu")
    assert str(device_utils.get_device()) == "cpu"
    assert device_utils.is_oom_error(RuntimeError("XPU out of memory"))


@pytest.mark.parametrize("env", [device_utils.ENV_DEVICE, device_utils.ENV_DEVICES])
@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_env_selection_sets_main_device_and_memory_limit(accelerators, monkeypatch, env, kind):
    # Explicit selection must not query memory on the auto-selected (CUDA) backend.
    accelerators["cuda"].is_available.return_value = True
    monkeypatch.setenv(env, kind + ":1" if env == device_utils.ENV_DEVICE else kind + ":1," + kind + ":0")
    init_gpu_config({"gpu_ind": [5, 6], "mem_limit": 2048})
    assert str(device_utils.get_device()) == kind + ":1"
    accelerators[kind].set_device.assert_called_once_with(torch.device(kind, 1))
    accelerators[kind].set_per_process_memory_fraction.assert_called_once_with(0.25, 1)


def test_env_list_takes_precedence_for_direct_potential_creation(accelerators, monkeypatch):
    monkeypatch.setenv(device_utils.ENV_DEVICE, "cpu")
    monkeypatch.setenv(device_utils.ENV_DEVICES, " xpu:1, xpu:0 ")
    assert str(device_utils.get_device()) == "xpu:1"
    assert [str(d) for d in get_devices()] == ["xpu:1", "xpu:0"]


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_memory_limit_uses_each_selected_device(accelerators, kind):
    api = accelerators[kind]
    api.is_available.return_value = True
    api.get_device_properties.side_effect = lambda d: SimpleNamespace(
        total_memory=(8192 if d.index == 0 else 1024) * 1024 * 1024)
    with pytest.raises(AssertionError, match="greater than total GPU memory"):
        init_gpu_config({"gpu_ind": [0, 1], "mem_limit": 2048})
    api.set_per_process_memory_fraction.assert_not_called()


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_memory_limit_without_device_index(accelerators, kind):
    apply_memory_limit(kind, 2048)
    accelerators[kind].set_per_process_memory_fraction.assert_called_once_with(0.25, 1)


def test_cpu_memory_limit_does_not_query_gpu_memory(accelerators, monkeypatch):
    monkeypatch.setenv(device_utils.ENV_DEVICE, "cpu")
    init_gpu_config({"mem_limit": 2048})
    for api in accelerators.values():
        api.get_device_properties.assert_not_called()
        api.set_per_process_memory_fraction.assert_not_called()


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_worker_sets_device_caps_memory_and_releases_cache(accelerators, monkeypatch, kind):
    conn = Mock()
    conn.recv.side_effect = [("release",), ("stop",)]
    monkeypatch.setattr(distributed, "Connection", Mock(return_value=conn))
    monkeypatch.setattr(device_utils, "set_num_threads", Mock())
    distributed._worker_main(123, 1, kind + ":1", 2048)
    api = accelerators[kind]
    api.set_device.assert_called_once_with(torch.device(kind, 1))
    api.set_per_process_memory_fraction.assert_called_once_with(0.25, 1)
    api.empty_cache.assert_called_once_with()
    assert conn.send.call_args_list[0].args == (("ok", 1),)
    assert conn.send.call_args_list[1].args == (("ok",),)
    conn.close.assert_called_once_with()


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_typed_and_forwarded_oom_errors(accelerators, kind):
    assert device_utils.is_oom_error(accelerators[kind].OutOfMemoryError("allocator exhausted"))
    assert device_utils.is_oom_error(distributed.WorkerError(kind.upper() + " out of memory"))
    assert not device_utils.is_oom_error(RuntimeError("unrelated failure"))


def test_level_zero_resource_exhaustion_is_treated_as_oom():
    # No "out of memory" substring; confirmed on real Intel XPU hardware to fire for genuine memory exhaustion
    # (a permanently resident test set/shard leaving no headroom for one more batch's forward/backward pass).
    assert device_utils.is_oom_error(
        RuntimeError("level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)")
    )
    assert device_utils.is_oom_error(RuntimeError("ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY"))
    assert not device_utils.is_oom_error(RuntimeError("ZE_RESULT_ERROR_DEVICE_LOST"))


# ------------------------------------------------------------------------------------------------
# ZE_FLAT_DEVICE_HIERARCHY tile grouping (a multi-tile XPU card exposed as several ``xpu:i`` indices)
# ------------------------------------------------------------------------------------------------
def _card_uuid(card, tile):
    """A fake UUID in the shape reported by a FLAT-hierarchy Intel XPU: one prefix per physical card, one
    distinct trailing segment per tile."""
    return "{}-0000-00000000000{}".format(card, tile)


def test_warns_when_a_physical_xpu_card_is_left_entirely_unused(accelerators, caplog):
    accelerators["xpu"].is_available.return_value = True
    accelerators["xpu"].device_count.return_value = 6
    # Three physical cards (two tiles each): aaaa=0/1, bbbb=2/3, cccc=4/5. Selecting only [0,1,2,3] doubles up
    # on aaaa and bbbb while cccc (both its tiles) is never touched - the real-world ZE_FLAT_DEVICE_HIERARCHY
    # pitfall, just scaled down.
    uuids = {0: _card_uuid("aaaa", 1), 1: _card_uuid("aaaa", 2), 2: _card_uuid("bbbb", 1),
             3: _card_uuid("bbbb", 2), 4: _card_uuid("cccc", 1), 5: _card_uuid("cccc", 2)}
    accelerators["xpu"].get_device_properties.side_effect = lambda d: SimpleNamespace(uuid=uuids[d.index])

    with caplog.at_level("WARNING"):
        get_devices({"gpu_ind": [0, 1, 2, 3]})
    assert any("physical GPU" in r.message and "not selected at all" in r.message for r in caplog.records)
    assert "xpu:0" in caplog.text and "xpu:1" in caplog.text


def test_no_warning_with_one_tile_per_physical_card(accelerators, caplog):
    accelerators["xpu"].is_available.return_value = True
    accelerators["xpu"].device_count.return_value = 4
    uuids = {0: _card_uuid("aaaa", 1), 1: _card_uuid("bbbb", 1), 2: _card_uuid("cccc", 1), 3: _card_uuid("dddd", 1)}
    accelerators["xpu"].get_device_properties.side_effect = lambda d: SimpleNamespace(uuid=uuids[d.index])

    with caplog.at_level("WARNING"):
        get_devices({"gpu_ind": [0, 1, 2, 3]})
    assert not any("not selected at all" in r.message for r in caplog.records)


def test_no_warning_when_every_visible_physical_card_is_selected(accelerators, caplog):
    """Deliberately using every tile of every card (e.g. all 12 tiles of a fully allocated node) is not a
    mistake: nothing else visible to this process is left unused."""
    accelerators["xpu"].is_available.return_value = True
    accelerators["xpu"].device_count.return_value = 4
    uuids = {0: _card_uuid("aaaa", 1), 1: _card_uuid("aaaa", 2), 2: _card_uuid("bbbb", 1), 3: _card_uuid("bbbb", 2)}
    accelerators["xpu"].get_device_properties.side_effect = lambda d: SimpleNamespace(uuid=uuids[d.index])

    with caplog.at_level("WARNING"):
        get_devices({"gpu_ind": [0, 1, 2, 3]})
    assert not any("not selected at all" in r.message for r in caplog.records)


def test_no_warning_for_a_single_device_or_without_uuid_support(accelerators, caplog):
    accelerators["xpu"].is_available.return_value = True
    with caplog.at_level("WARNING"):
        get_devices({"gpu_ind": 0})  # a single device: nothing to compare
        get_devices({"gpu_ind": [0, 1]})  # mocked get_device_properties has no `uuid` attribute by default
    assert not any("not selected at all" in r.message for r in caplog.records)
