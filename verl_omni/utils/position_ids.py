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
"""Normalize nested position tensors at worker input boundaries."""

from collections.abc import MutableMapping
from typing import Any

import torch


def normalize_nested_position_ids(data: MutableMapping[str, Any]) -> None:
    """Keep packed 3D jagged position IDs in ``(batch, axes, tokens)`` layout.

    Args:
        data: Batch whose integer ``position_ids`` field is replaced in place.

    Raises:
        ValueError: The nested layout is invalid or cannot be identified safely.
    """
    positions = data.get("position_ids")
    if not isinstance(positions, torch.Tensor) or not positions.is_nested or positions.dim() != 3:
        return
    if positions.layout != torch.jagged:
        raise ValueError("3D nested position_ids must use torch.jagged layout.")
    if positions.lengths() is not None:
        raise ValueError("Nested position_ids must be packed; layouts with explicit lengths are unsupported.")

    values, offsets = positions.values(), positions.offsets()
    lengths = offsets.diff()
    if offsets[0].item() != 0 or (lengths < 0).any().item():
        raise ValueError("Nested position_ids offsets must start at zero and be nondecreasing.")
    total_length = offsets[-1].item()
    axis_one = total_length == values.shape[0]
    axis_two = total_length == values.shape[1]
    canonical = (
        positions._ragged_idx == 2 and isinstance(positions.shape[1], int) and positions.shape[1] == values.shape[0]
    )
    input_ids = data.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.is_nested and input_ids.dim() == 2:
        token_lengths = input_ids.offsets().diff().to(device=offsets.device)
        if len(token_lengths) != len(lengths):
            raise ValueError("Nested input_ids and position_ids must have the same batch size.")
        axis_one = axis_one and (token_lengths == values.shape[1]).all().item()
        axis_two = axis_two and torch.equal(token_lengths, lengths)
        # Token lengths distinguish square packed buffers without guessing their axes.
        if axis_two:
            axis_one = False

    if axis_two and (not axis_one or canonical):
        if canonical:
            return
        # Serialization can lose the ragged-axis metadata while preserving values and offsets.
        data["position_ids"] = torch.nested.nested_tensor_from_jagged(values, offsets, jagged_dim=2)
    elif axis_one and not axis_two:
        if not len(lengths) or lengths[0].item() <= 0 or (lengths != lengths[0]).any().item():
            raise ValueError("Nested position_ids must have the same number of position axes in every sample.")
        # Equal-shaped rows make automatic jagged construction choose the axes dimension.
        rows = values.split(lengths.tolist(), dim=0)
        token_offsets = torch.arange(len(rows) + 1, dtype=offsets.dtype, device=offsets.device) * values.shape[1]
        data["position_ids"] = torch.nested.nested_tensor_from_jagged(
            torch.cat(rows, dim=1), token_offsets, jagged_dim=2
        )
    else:
        raise ValueError("Cannot identify the ragged axis of nested position_ids from values and offsets.")
