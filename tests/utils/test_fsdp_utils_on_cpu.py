# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for diffusion FSDP LoRA param collection."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

from verl_omni.utils.fsdp_utils import collect_lora_params


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.to_q(x)


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block()])

    def forward(self, x):
        return self.transformer_blocks[0](x)


def _peft_dit():
    model = get_peft_model(
        _TinyDiT(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none"),
    )
    model.eval()
    return model


def test_fsdp2_non_layered_collects_lora_without_child_units(monkeypatch):
    """SD3.5 v1 uses FSDP2 + layered_summon=False; child-unit PEFT lookup is empty."""
    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 2)
    monkeypatch.setattr(fsdp_utils, "_iter_fsdp2_submodules", lambda _: iter(()))

    params = collect_lora_params(
        module,
        layered_summon=False,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert any("lora_" in name for name in params)
    assert all(".default." not in name for name in params)
    assert all(isinstance(t, torch.Tensor) for t in params.values())


def test_layered_diffusers_falls_back_when_prefix_walker_is_empty(monkeypatch):
    """Qwen-Image e2e: layered_summon=True; prefix walker skips non-block FSDP units."""
    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()

    def _version(m):
        return 2 if m is module else 0

    monkeypatch.setattr(fsdp_utils, "fsdp_version", _version)

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert any("lora_" in name for name in params)
    assert all(".default." not in name for name in params)
    assert all(isinstance(t, torch.Tensor) for t in params.values())


def test_layered_collects_fsdp_leaf_lora_when_peft_dump_is_empty(monkeypatch):
    """Qwen-Image-Edit: named targets + FSDP1 leaf wrap; PEFT prefix dump is {}."""
    from collections import OrderedDict
    from contextlib import nullcontext

    import verl_omni.utils.fsdp_utils as fsdp_utils

    class _Adapter(nn.Module):
        def __init__(self, fill: float):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2, 4), fill))

    class _LoraA(nn.Module):
        def __init__(self):
            super().__init__()
            self.default = _Adapter(1.0)
            self.old = _Adapter(2.0)

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = _LoraA()

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attn()

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([_Block()])

    module = _Tiny()

    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda m: 1 if m is module else 0)
    monkeypatch.setattr(fsdp_utils, "_peft_lora_params_to_cpu", lambda *args, **kwargs: OrderedDict())
    monkeypatch.setattr(
        "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert list(params) == ["transformer_blocks.0.attn.lora_A.weight"]
    torch.testing.assert_close(params["transformer_blocks.0.attn.lora_A.weight"], torch.ones(2, 4))


def test_layered_diffusers_falls_back_for_frozen_non_default_adapter(monkeypatch):
    """DiffusionNFT syncs a frozen "old" adapter; PEFT's own state-dict lookup misses it."""
    from contextlib import nullcontext

    import verl_omni.utils.fsdp_utils as fsdp_utils

    module = _peft_dit()
    module.add_adapter("old", LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none"))

    def _version(m):
        return 2 if m is module else 0

    monkeypatch.setattr(fsdp_utils, "fsdp_version", _version)
    # Simulate PEFT's adapter-name lookup missing the frozen adapter's tensors even
    # though they are present in the submodule's own state dict / named_parameters.
    monkeypatch.setattr(fsdp_utils, "get_peft_model_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params",
        lambda *args, **kwargs: nullcontext(),
    )

    params = collect_lora_params(
        module,
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
        adapter_name="old",
    )
    assert any("lora_" in name and ".old." not in name for name in params)
    assert all(isinstance(t, torch.Tensor) for t in params.values())


def test_collect_strips_nested_fsdp_wrapper_tokens_for_vllm(monkeypatch):
    """Qwen-Image e2e: nested FSDP wraps leak into PEFT dump keys; vLLM rejects them."""
    from collections import OrderedDict

    import verl_omni.utils.fsdp_utils as fsdp_utils

    dirty = "transformer_blocks.0._fsdp_wrapped_module.attn.to_q.lora_A.default._fsdp_wrapped_module.weight"
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _: 1)
    monkeypatch.setattr(
        fsdp_utils,
        "_layered_summon_lora_params_diffusers",
        lambda *args, **kwargs: OrderedDict({dirty: torch.ones(2, 4)}),
    )

    params = collect_lora_params(
        nn.Linear(4, 4, bias=False),
        layered_summon=True,
        base_sync_done=True,
        is_diffusers=True,
    )
    assert list(params) == ["transformer_blocks.0.attn.to_q.lora_A.weight"]
    assert all("_fsdp_wrapped_module" not in name for name in params)
    assert all(".default." not in name for name in params)


@pytest.mark.parametrize("use_orig_params", [False, True])
def test_fsdp1_non_layered_collects_all_nested_adapters_on_cpu(tmp_path, monkeypatch, use_orig_params):
    """Nested FSDP state-dict names must not drop selected-adapter tensors."""
    model = _TinyDiT()
    model.transformer_blocks.append(_Block())
    model.to_q = nn.Linear(4, 4, bias=False)
    lora_config = LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"], bias="none")
    peft_model = get_peft_model(model, lora_config)
    peft_model.add_adapter("old", lora_config)
    peft_model.set_adapter("default")

    with torch.no_grad():
        for index, (name, param) in enumerate(peft_model.named_parameters()):
            if "lora_" in name:
                param.fill_(index + 1)
    expected = {
        adapter: {
            name: param.detach().clone()
            for name, param in get_peft_model_state_dict(peft_model, adapter_name=adapter).items()
        }
        for adapter in ("default", "old")
    }
    assert len(expected["default"]) == 6
    assert expected["default"].keys() == expected["old"].keys()

    device_module = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setattr("verl.utils.device.get_torch_device", lambda: device_module)
    monkeypatch.setattr("verl.utils.fsdp_utils.get_torch_device", lambda: device_module)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'dist_init'}", rank=0, world_size=1)
    try:
        fsdp_model = FSDP(
            peft_model,
            auto_wrap_policy=get_fsdp_wrap_policy(
                peft_model, config={"transformer_layer_cls_to_wrap": ["_Block"]}, is_lora=True
            ),
            device_id=torch.device("cpu"),
            use_orig_params=use_orig_params,
        )
        assert isinstance(model.transformer_blocks[0], FSDP)
        assert isinstance(model.transformer_blocks[0].to_q.lora_A["default"], FSDP)
        for adapter in ("default", "old", "default"):
            params = collect_lora_params(
                fsdp_model,
                layered_summon=False,
                base_sync_done=True,
                is_diffusers=True,
                adapter_name=adapter,
            )
            assert params.keys() == expected[adapter].keys()
            for name, tensor in params.items():
                torch.testing.assert_close(tensor, expected[adapter][name])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("saved_module", ["batch_norm", "linear", "embedding"])
def test_fsdp1_non_layered_preserves_peft_saved_state(tmp_path, monkeypatch, saved_module):
    """PEFT exports keep saved-module buffers, biases, and embedding adapters."""
    model = nn.Module()
    model.to_q = nn.Linear(4, 4)
    if saved_module == "batch_norm":
        model.saved = nn.BatchNorm1d(4)
    elif saved_module == "linear":
        model.saved = nn.Linear(4, 2)
    else:
        model.to_q = nn.Embedding(8, 4)
        model.saved = nn.Linear(4, 2)
    model.saved.register_buffer("scratch", torch.ones(1), persistent=False)
    peft_model = get_peft_model(model, LoraConfig(r=2, target_modules=["to_q"], modules_to_save=["saved"], bias="all"))
    peft_model.add_adapter("old", LoraConfig(r=2, target_modules=["to_q"], modules_to_save=["saved"], bias="none"))
    peft_model.set_adapter("default")
    with torch.no_grad():
        for index, tensor in enumerate((*peft_model.parameters(), *peft_model.buffers())):
            tensor.fill_(index + 1)
    expected = {
        adapter: {
            name: tensor.clone() for name, tensor in get_peft_model_state_dict(peft_model, adapter_name=adapter).items()
        }
        for adapter in ("default", "old")
    }
    assert all("scratch" not in name for params in expected.values() for name in params)
    if saved_module == "batch_norm":
        for params in expected.values():
            assert {name.removeprefix("base_model.model.saved.") for name in params if name.count(".") == 3} == {
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            }
    elif saved_module == "embedding":
        for params in expected.values():
            assert {name.rsplit(".", 1)[-1] for name in params if "lora_embedding_" in name} == {
                "lora_embedding_A",
                "lora_embedding_B",
            }
    else:
        assert "base_model.model.to_q.base_layer.bias" in expected["default"]
        assert "base_model.model.saved.weight" in expected["default"]
        assert "base_model.model.saved.bias" in expected["default"]

    device_module = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setattr("verl.utils.device.get_torch_device", lambda: device_module)
    monkeypatch.setattr("verl.utils.fsdp_utils.get_torch_device", lambda: device_module)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'dist_init'}", rank=0, world_size=1)
    try:
        fsdp_model = FSDP(
            peft_model,
            auto_wrap_policy=get_fsdp_wrap_policy(peft_model, is_lora=True),
            device_id=torch.device("cpu"),
            use_orig_params=True,
        )
        for adapter in ("default", "old", "default"):
            params = collect_lora_params(fsdp_model, layered_summon=False, base_sync_done=True, adapter_name=adapter)
            assert params.keys() == expected[adapter].keys()
            assert all(".default." not in name and ".old." not in name for name in params)
            for name, tensor in params.items():
                torch.testing.assert_close(tensor, expected[adapter][name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
