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
"""Restore packed multi-axis position IDs at training-worker boundaries."""

from collections.abc import MutableMapping
from typing import Any

import torch


def normalize_nested_position_ids(data: MutableMapping[str, Any]) -> None:
    """Restore ``(batch, axes, tokens)`` positions without changing their values.

    Equal-length samples can be packed along the axes dimension by automatic
    jagged construction. Serialization can also lose the ragged-axis metadata.
    Token offsets identify both layouts before the worker selects minibatches.

    Args:
        data: Packed token batch with integer ``input_ids`` and ``position_ids``.

    Raises:
        ValueError: Multi-axis positions do not match the packed token lengths.
    """
    positions = data.get("position_ids")
    if not isinstance(positions, torch.Tensor) or not positions.is_nested or positions.dim() != 3:
        return
    input_ids = data.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or not input_ids.is_nested or input_ids.dim() != 2:
        raise ValueError("Nested multi-axis position_ids require packed 2D input_ids.")
    if positions.layout != torch.jagged or input_ids.layout != torch.jagged:
        raise ValueError("Nested position_ids and input_ids must use torch.jagged layout.")
    if positions.lengths() is not None or input_ids.lengths() is not None:
        raise ValueError("Nested position_ids and input_ids must be packed without explicit lengths.")

    values, offsets = positions.values(), positions.offsets()
    token_offsets = input_ids.offsets().to(offsets)
    lengths, token_lengths = offsets.diff(), token_offsets.diff()
    if len(lengths) != len(token_lengths):
        raise ValueError("Nested position_ids and input_ids must have the same batch size.")
    if offsets[0].item() != 0 or token_offsets[0].item() != 0 or (lengths < 0).any().item():
        raise ValueError("Nested position_ids and input_ids must have packed nondecreasing offsets.")

    if values.shape[1] == token_offsets[-1].item() and torch.equal(lengths, token_lengths):
        # Reconstruct the tensor: changing _ragged_idx alone leaves stale shape metadata.
        packed_values = values
    elif (
        len(lengths) > 0
        and offsets[-1].item() == values.shape[0]
        and lengths[0].item() > 0
        and (lengths == lengths[0]).all().item()
        and (token_lengths == values.shape[1]).all().item()
    ):
        packed_values = torch.cat(values.split(lengths.tolist(), dim=0), dim=1)
    else:
        raise ValueError("Nested position_ids do not match the packed input_ids token lengths.")

    data["position_ids"] = torch.nested.nested_tensor_from_jagged(packed_values, token_offsets, jagged_dim=2)
