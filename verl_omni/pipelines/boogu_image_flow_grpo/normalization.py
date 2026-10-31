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

"""Preserve Boogu's empty reference-token gradients without fused NPU RMSNorm."""

import torch
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch import nn


def _is_npu_tensor(hidden_states: torch.Tensor) -> bool:
    return hidden_states.device.type == "npu"


def _is_empty_normalized_input(hidden_states: torch.Tensor, shape: tuple[int, ...]) -> bool:
    return (
        _is_npu_tensor(hidden_states)
        and hidden_states.numel() == 0
        and hidden_states.is_floating_point()
        and len(shape) > 0
        and hidden_states.ndim >= len(shape)
        and hidden_states.shape[-len(shape) :] == shape
    )


class _BooguTorchRMSNorm(nn.RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if _is_empty_normalized_input(hidden_states, self.normalized_shape):
            if self.weight is None:
                return hidden_states
            # Keep zero weight gradients and the native input-dtype output.
            return (hidden_states * self.weight).to(hidden_states.dtype)
        return super().forward(hidden_states)


class _BooguDiffusersRMSNorm(DiffusersRMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.weight is None or not _is_empty_normalized_input(hidden_states, self.dim):
            return super().forward(hidden_states)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        # The native fused output retains its input dtype before bias addition.
        hidden_states = (hidden_states * self.weight).to(hidden_states.dtype)
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
        return hidden_states


def install_empty_safe_rms_norm(ref_image_refiner: nn.Module) -> None:
    """Replace reference-refiner norm leaves before PEFT/FSDP, reusing parameters."""
    for parent in ref_image_refiner.modules():
        for name, norm in list(parent.named_children()):
            if type(norm) is nn.RMSNorm:
                replacement = _BooguTorchRMSNorm(
                    norm.normalized_shape, eps=norm.eps, elementwise_affine=norm.elementwise_affine
                )
            elif type(norm) is DiffusersRMSNorm:
                replacement = _BooguDiffusersRMSNorm(
                    norm.dim, eps=norm.eps, elementwise_affine=norm.elementwise_affine, bias=norm.bias is not None
                )
                replacement.bias = norm.bias
            else:
                continue
            replacement.weight = norm.weight
            replacement.train(norm.training)
            setattr(parent, name, replacement)
