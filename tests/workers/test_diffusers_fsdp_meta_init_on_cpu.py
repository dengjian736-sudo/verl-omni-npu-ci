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
"""CPU tests for meta-rank config-only init in the diffusers FSDP engine."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate import init_empty_weights

import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


class _FakeDiffusersModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.config = SimpleNamespace()
        self.attn_backend = None

    def set_attention_backend(self, backend):
        self.attn_backend = backend


def _make_engine(**overrides):
    engine = object.__new__(PPODiffusersFSDPEngine)
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "config_path", None)
    object.__setattr__(model_config, "local_path", "/fake/ckpt")
    object.__setattr__(model_config, "trust_remote_code", True)
    object.__setattr__(model_config, "transformer_subfolder", "transformer")
    object.__setattr__(model_config, "attn_backend", "flash_attn")
    object.__setattr__(model_config, "enable_gradient_checkpointing", False)
    for key, value in overrides.items():
        object.__setattr__(model_config, key, value)
    engine.model_config = model_config
    engine.engine_config = MagicMock(model_dtype=None, forward_only=False)
    engine.device_mesh = None
    return engine


def _build_default_path(engine, *, init_context, loaded_module):
    adapter_cls = MagicMock()
    adapter_cls.preserve_fp32_modules.return_value = False

    with (
        patch.object(engine, "_build_module_from_registry", return_value=None),
        patch.object(diffusers_impl, "get_init_weight_context_manager", return_value=init_context),
        patch.object(diffusers_impl.DiffusionModelBase, "get_class", return_value=adapter_cls),
        patch("diffusers.AutoModel.from_config", return_value=loaded_module) as mock_from_config,
        patch("diffusers.AutoModel.from_pretrained", return_value=loaded_module) as mock_from_pretrained,
        patch("verl.utils.torch_dtypes.PrecisionType") as precision_type,
    ):
        precision_type.to_dtype.side_effect = lambda value: value
        result = engine._build_module()
    return result, mock_from_config, mock_from_pretrained


def test_build_module_on_meta_rank_builds_from_config_without_reading_the_checkpoint():
    engine = _make_engine()
    loaded = _FakeDiffusersModule()

    result, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=init_empty_weights, loaded_module=loaded
    )

    mock_from_pretrained.assert_not_called()
    mock_from_config.assert_called_once_with(
        engine.model_config.local_path,
        trust_remote_code=True,
        subfolder="transformer",
    )
    assert result is loaded
    assert loaded.attn_backend == "flash_attn"
    assert loaded.proj.weight.dtype == torch.float32
    assert callable(loaded.can_generate)
    assert callable(loaded.config.save_pretrained)


def test_build_module_on_rank0_still_loads_from_pretrained():
    engine = _make_engine()
    loaded = _FakeDiffusersModule()

    result, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=nullcontext, loaded_module=loaded
    )

    mock_from_config.assert_not_called()
    mock_from_pretrained.assert_called_once_with(
        engine.model_config.local_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        subfolder="transformer",
    )
    assert result is loaded
    assert loaded.attn_backend == "flash_attn"
    assert loaded.proj.weight.dtype == torch.float32


def test_build_module_on_meta_rank_uses_config_path_when_set():
    engine = _make_engine(config_path="/fake/config", transformer_subfolder="transformer")
    loaded = _FakeDiffusersModule()

    _, mock_from_config, mock_from_pretrained = _build_default_path(
        engine, init_context=init_empty_weights, loaded_module=loaded
    )

    mock_from_pretrained.assert_not_called()
    mock_from_config.assert_called_once_with(
        "/fake/config",
        trust_remote_code=True,
        subfolder="",
    )


def test_meta_rank_adapters_beyond_the_first_match_base_dtype_and_device():
    """Multi-adapter LoRA on a meta-built module must stay dtype-uniform for FSDP.

    peft applies the base layer's device/dtype only when it first wraps a module;
    adapters added later are created on CPU in the default dtype, which mixes
    dtypes inside FSDP wrapping units once the base module stays on meta.
    """
    from diffusers import QwenImageTransformer2DModel
    from peft import LoraConfig

    from verl_omni.workers.engine.lora_adapter_mixin import cast_adapter_params_to_base

    config = {
        "patch_size": 2,
        "in_channels": 64,
        "out_channels": 16,
        "num_layers": 2,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "guidance_embeds": False,
        "axes_dims_rope": (4, 6, 6),
    }
    with init_empty_weights():
        module = QwenImageTransformer2DModel.from_config(config).to(torch.bfloat16)

    lora = LoraConfig(r=4, lora_alpha=8, init_lora_weights=False, target_modules="all-linear")
    module.add_adapter(lora, adapter_name="default")
    module.add_adapter(lora, adapter_name="old")

    cast_adapter_params_to_base(module)

    for name, param in module.named_parameters():
        assert param.dtype == torch.bfloat16, name
        assert param.device.type == "meta", name


def test_real_h3_config_only_rank_matches_pretrained_precision(tmp_path):
    from diffusers import MiniMaxH3Transformer3DModel

    import verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter  # noqa: F401

    checkpoint = tmp_path / "h3"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        MiniMaxH3Transformer3DModel(
            in_channels=8,
            audio_in_channels=8,
            num_layers=1,
            num_refiner_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            attention_head_dim=32,
            ffn_dim=128,
            text_dim=32,
            freq_dim=64,
            time_embed_hidden_dim=64,
            time_embed_dim=32,
            rope_freq_dim=4,
        ).save_pretrained(checkpoint)

    original_dtype = torch.get_default_dtype()
    maps, orders, buffers = [], [], []
    for context in (lambda: torch.device("cpu"), init_empty_weights):
        engine = _make_engine(
            config_path=str(checkpoint),
            local_path=str(checkpoint),
            architecture="MiniMaxH3Pipeline",
            algorithm="flow_grpo",
            attn_backend="native",
            trust_remote_code=False,
            lora_rank=8,
            lora_alpha=16,
            lora_init_weights="gaussian",
            policy_state_adapters=["default"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
            target_parameters=None,
            exclude_modules=None,
            lora_dtype=None,
            lora_adapter_path=None,
            lora={"merge": False},
        )
        engine.engine_config = SimpleNamespace(model_dtype="bfloat16", forward_only=False)
        with patch.object(diffusers_impl, "get_init_weight_context_manager", return_value=context):
            module = engine._build_module()
        for name, param in module.named_parameters():
            keep_fp32 = any(part in name.split(".") for part in module._keep_in_fp32_modules)
            expected = torch.float32 if keep_fp32 else torch.bfloat16
            assert param.dtype == expected, name
        module = engine._build_lora_module(module)
        parameters = dict(module.named_parameters())
        assert any("lora_A.default" in name for name in parameters)
        assert all(param.is_meta == (context is init_empty_weights) for param in parameters.values())
        maps.append({name: (tuple(param.shape), param.dtype) for name, param in parameters.items()})
        orders.append(list(parameters))
        buffers.append({name: (tuple(value.shape), value.dtype) for name, value in module.named_buffers()})
        assert dict(module.named_buffers())["rope.inv_freq"].dtype == torch.float32
        assert torch.get_default_dtype() == original_dtype
    assert maps[0] == maps[1]
    assert orders[0] == orders[1]
    assert buffers[0] == buffers[1]


def test_config_only_cast_preserves_aliases_grad_and_computed_buffers():
    class Module(torch.nn.Module):
        _keep_in_fp32_modules = ["keep"]

        def __init__(self):
            super().__init__()
            self.keep = torch.nn.Linear(2, 2, dtype=torch.float64)
            self.keep.weight.requires_grad_(False)
            self.keep.alias = self.keep.weight
            self.not_keep = torch.nn.Linear(2, 2)
            self.keep.register_buffer("saved", torch.ones(2, dtype=torch.float64))
            self.keep.register_buffer("computed", torch.ones(2, dtype=torch.float64), persistent=False)
            self.register_buffer("index", torch.ones(2, dtype=torch.int64))
            self.register_buffer("computed_default", torch.ones(2), persistent=False)

    original_dtype = torch.get_default_dtype()
    with patch("diffusers.AutoModel.from_config", side_effect=lambda *args, **kwargs: Module()):
        module = diffusers_impl._build_diffusers_module_from_config("unused", torch.bfloat16)
    assert module.keep.weight is module.keep.alias
    assert not module.keep.weight.requires_grad
    assert module.keep.weight.dtype == module.keep.saved.dtype == torch.float32
    assert module.not_keep.weight.dtype == module.computed_default.dtype == torch.bfloat16
    assert module.keep.computed.dtype == torch.float64
    assert module.index.dtype == torch.int64
    assert torch.get_default_dtype() == original_dtype


def test_config_only_constructor_restores_default_dtype_on_error():
    original_dtype = torch.get_default_dtype()
    with patch("diffusers.AutoModel.from_config", side_effect=RuntimeError("constructor failed")):
        with pytest.raises(RuntimeError, match="constructor failed"):
            diffusers_impl._build_diffusers_module_from_config("unused", torch.bfloat16)
    assert torch.get_default_dtype() == original_dtype
