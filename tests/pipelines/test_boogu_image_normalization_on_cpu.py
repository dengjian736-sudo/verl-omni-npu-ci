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
"""Preserve Boogu empty-refiner norm contracts using small CPU modules."""

import re
import sys
from itertools import product
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers.models import normalization as diffusers_normalization
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch import nn

from verl_omni.pipelines.boogu_image_flow_grpo import normalization

_DTYPE_PAIRS = [
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.float16, torch.float32),
]


@pytest.fixture(autouse=True)
def cpu_reference_backend(monkeypatch):
    """Keep native CPU references on CPU even when torch_npu is installed."""
    monkeypatch.setattr(diffusers_normalization, "is_torch_npu_available", lambda: False)


def _installed(norm):
    refiner = nn.Sequential(norm)
    normalization.install_empty_safe_rms_norm(refiner)
    return refiner[0]


def _observe(norm, inputs):
    targets = {"input": inputs, **dict(norm.named_parameters())}
    targets = {name: tensor for name, tensor in targets.items() if tensor.requires_grad}
    output = norm(inputs)
    gradients = torch.autograd.grad(output.sum(), tuple(targets.values())) if targets else ()
    return output, dict(zip(targets, gradients, strict=True))


def _assert_same_observation(actual, expected):
    actual_output, actual_gradients = actual
    expected_output, expected_gradients = expected
    assert actual_output.requires_grad == expected_output.requires_grad
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in actual_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name], rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype,weight_dtype", _DTYPE_PAIRS)
@pytest.mark.parametrize("input_grad,weight_grad", product((False, True), repeat=2))
def test_torch_empty_matches_native_output_and_gradients(
    monkeypatch, input_dtype, weight_dtype, input_grad, weight_grad
):
    """Empty outputs retain native dtype and zero, rather than absent, weight gradients."""
    native = nn.RMSNorm((2, 2), eps=1e-5, dtype=weight_dtype)
    native.weight.requires_grad_(weight_grad)
    portable = _installed(native)
    monkeypatch.setattr(normalization, "_is_npu_tensor", lambda tensor: True)
    expected_input = torch.empty(2, 0, 2, 2, dtype=input_dtype, requires_grad=input_grad)
    actual_input = torch.empty_like(expected_input, requires_grad=input_grad)
    expected = _observe(native, expected_input)
    actual = _observe(portable, actual_input)
    assert actual[0].shape == expected_input.shape
    assert actual[0].dtype == input_dtype
    _assert_same_observation(actual, expected)
    if weight_grad:
        torch.testing.assert_close(actual[1]["weight"], torch.zeros_like(native.weight), rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype,weight_dtype", _DTYPE_PAIRS)
@pytest.mark.parametrize(
    "with_bias,input_grad,weight_grad,bias_grad",
    [
        (with_bias, input_grad, weight_grad, bias_grad)
        for with_bias in (False, True)
        for input_grad, weight_grad in product((False, True), repeat=2)
        for bias_grad in ((False, True) if with_bias else (False,))
    ],
)
def test_diffusers_empty_preserves_npu_dtype_and_zero_gradients(
    monkeypatch, input_dtype, weight_dtype, with_bias, input_grad, weight_grad, bias_grad
):
    """The measured NPU pre-cast contract precedes optional FP32 bias promotion."""
    native = DiffusersRMSNorm(4, eps=1e-5, bias=with_bias).to(weight_dtype)
    native.weight.requires_grad_(weight_grad)
    if with_bias:
        native.bias = nn.Parameter(torch.arange(4, dtype=torch.float32), requires_grad=bias_grad)
    portable = _installed(native)
    monkeypatch.setattr(normalization, "_is_npu_tensor", lambda tensor: True)
    inputs = torch.empty(1, 0, 2, 4, dtype=input_dtype, requires_grad=input_grad)
    output, gradients = _observe(portable, inputs)
    kernel_dtype = weight_dtype if weight_dtype in (torch.float16, torch.bfloat16) else input_dtype
    expected_dtype = torch.promote_types(kernel_dtype, torch.float32) if with_bias else kernel_dtype
    assert output.shape == inputs.shape
    assert output.dtype == expected_dtype
    assert output.requires_grad == (input_grad or weight_grad or (with_bias and bias_grad))
    expected_targets = {"input": inputs, **dict(portable.named_parameters())}
    assert set(gradients) == {name for name, tensor in expected_targets.items() if tensor.requires_grad}
    for name, gradient in gradients.items():
        target = expected_targets[name]
        assert gradient.shape == target.shape
        assert gradient.dtype == target.dtype
        torch.testing.assert_close(gradient, torch.zeros_like(target), rtol=0, atol=0)


@pytest.mark.parametrize("norm_type", [nn.RMSNorm, DiffusersRMSNorm])
@pytest.mark.parametrize(
    "shape,simulate_npu",
    [
        pytest.param((2, 0, 4), False, id="cpu-empty"),
        pytest.param((2, 3, 4), True, id="nonempty"),
        pytest.param((2, 0, 1), True, id="invalid-normalized-shape"),
    ],
)
def test_other_inputs_preserve_original_behavior(monkeypatch, norm_type, shape, simulate_npu):
    """CPU, populated, and incompatible shapes keep their parent's result or error."""
    native = norm_type(4, eps=1e-5)
    portable = _installed(native)
    if simulate_npu:
        monkeypatch.setattr(normalization, "_is_npu_tensor", lambda tensor: True)
    reference_input = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_()
    try:
        expected = _observe(native, reference_input)
    except RuntimeError as error:
        with pytest.raises(type(error), match=re.escape(str(error))):
            _observe(portable, actual_input)
    else:
        _assert_same_observation(_observe(portable, actual_input), expected)


@pytest.mark.parametrize("norm_type", [nn.RMSNorm, DiffusersRMSNorm])
@pytest.mark.parametrize("shape", [(2, 0, 4), (2, 3, 4)])
def test_without_affine_preserves_parent_contract(monkeypatch, norm_type, shape):
    """Diffusers without affine stays on its original forward, including empty inputs."""
    native = norm_type(4, eps=1e-5, elementwise_affine=False)
    portable = _installed(native)
    monkeypatch.setattr(normalization, "_is_npu_tensor", lambda tensor: True)
    reference_input = torch.randn(shape, requires_grad=True)
    actual_input = reference_input.detach().clone().requires_grad_()
    actual = _observe(portable, actual_input)
    _assert_same_observation(actual, _observe(native, reference_input))
    if norm_type is DiffusersRMSNorm:
        assert actual[0] is not actual_input


def test_installer_preserves_parameters_and_only_changes_requested_refiner():
    """Nested installation retains parameter identities, state, configuration and idempotency."""

    class CustomNorm(nn.RMSNorm):
        pass

    torch_norm = nn.RMSNorm((2, 2), eps=2e-5, dtype=torch.bfloat16).eval()
    torch_norm.weight.requires_grad_(False)
    diffusers_norm = DiffusersRMSNorm(4, eps=3e-4, bias=True).to(torch.float16)
    diffusers_norm.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
    custom = CustomNorm(4)
    refiner = nn.ModuleDict({"torch": torch_norm, "nested": nn.Sequential(diffusers_norm), "custom": custom})
    outside = nn.RMSNorm(4)
    model = nn.ModuleDict({"refiner": refiner, "outside": outside})
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    normalization.install_empty_safe_rms_norm(refiner)

    assert model["outside"] is outside
    assert refiner["custom"] is custom
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
    assert model.state_dict().keys() == state.keys()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    for name, original in (("torch", torch_norm), ("nested.0", diffusers_norm)):
        replacement = refiner.get_submodule(name)
        assert replacement is not original
        assert replacement.weight is original.weight
        assert replacement.training == original.training
        assert replacement.eps == original.eps
        assert replacement.elementwise_affine == original.elementwise_affine
    assert refiner["torch"].normalized_shape == (2, 2)
    assert refiner["nested"][0].dim == (4,)
    assert refiner["nested"][0].bias is diffusers_norm.bias
    module_ids = {name: id(module) for name, module in model.named_modules()}
    normalization.install_empty_safe_rms_norm(refiner)
    assert {name: id(module) for name, module in model.named_modules()} == module_ids


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
@pytest.mark.parametrize("use_nft", [False, True], ids=["flow-grpo", "nft"])
def test_build_module_installs_only_reference_refiner_on_npu(monkeypatch, device, use_nft):
    """The inherited production loader applies the installer only to NPU reference tokens."""
    from verl_omni.pipelines.boogu_image_diffusion_nft.diffusers_training_adapter import BooguImageDiffusionNFT
    from verl_omni.pipelines.boogu_image_flow_grpo import diffusers_training_adapter as adapter
    from verl_omni.workers.config import DiffusionModelConfig

    module = nn.Module()
    module.rope_embedder = SimpleNamespace(
        theta=10000, axes_dim=(4, 2, 2), axes_lens=(2, 2, 2), patch_size=2, training=False
    )
    module.ref_image_refiner = nn.Sequential(nn.RMSNorm(4), DiffusersRMSNorm(4, eps=1e-5))
    outside = module.outside = nn.RMSNorm(4)
    loader = Mock(return_value=module)
    transformer_module = ModuleType("boogu.models.transformers.transformer_boogu")
    transformer_module.BooguImageTransformer2DModel = SimpleNamespace(from_pretrained=loader)
    rotary_module = ModuleType(f"{adapter.__package__}.rotary_embedding")
    rotary_constructor = Mock(return_value=nn.Identity())
    rotary_module.BooguImagePortableRotaryPosEmbed = rotary_constructor
    monkeypatch.setitem(sys.modules, transformer_module.__name__, transformer_module)
    monkeypatch.setitem(sys.modules, rotary_module.__name__, rotary_module)
    installer = Mock(wraps=normalization.install_empty_safe_rms_norm)
    monkeypatch.setattr(adapter, "get_device_name", lambda: device)
    monkeypatch.setattr(normalization, "install_empty_safe_rms_norm", installer)
    config = object.__new__(DiffusionModelConfig)
    for name, value in {
        "config_path": None,
        "local_path": "unused-checkpoint",
        "transformer_subfolder": "transformer",
    }.items():
        object.__setattr__(config, name, value)
    adapter_class = BooguImageDiffusionNFT if use_nft else adapter.BooguImage

    assert adapter_class.build_module(config, torch.bfloat16) is module

    loader.assert_called_once_with("unused-checkpoint", subfolder="transformer", torch_dtype=torch.bfloat16)
    rotary_constructor.assert_called_once_with(theta=10000, axes_dim=(4, 2, 2), axes_lens=(2, 2, 2), patch_size=2)
    assert BooguImageDiffusionNFT.build_module.__func__ is adapter.BooguImage.build_module.__func__
    assert module.outside is outside
    if device == "npu":
        installer.assert_called_once_with(module.ref_image_refiner)
    else:
        installer.assert_not_called()
        assert type(module.ref_image_refiner[0]) is nn.RMSNorm
        assert type(module.ref_image_refiner[1]) is DiffusersRMSNorm
