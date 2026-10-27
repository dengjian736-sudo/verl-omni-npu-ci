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
"""Regression tests for jagged position IDs reconstructed across worker boundaries."""

import importlib.util
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

_PATH = Path(__file__).parents[2] / "verl_omni/utils/position_ids.py"
_SPEC = importlib.util.spec_from_file_location("position_ids_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
normalize_nested_position_ids = _MODULE.normalize_nested_position_ids


@pytest.mark.parametrize("lengths", [(187, 187), (87, 100), (187,)])
@pytest.mark.parametrize("stale_axis", [False, True])
def test_nested_position_ids_preserve_every_sample(lengths, stale_axis):
    rows = [torch.arange(4 * length).reshape(4, length) + index for index, length in enumerate(lengths)]
    positions = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    if stale_axis:
        positions._ragged_idx = 3 - positions._ragged_idx
    data = {"position_ids": positions}

    normalize_nested_position_ids(data)

    actual = data["position_ids"]
    assert actual._ragged_idx == 2
    assert actual.shape[1] == 4
    assert actual.offsets().diff().tolist() == list(lengths)
    for expected, sample in zip(rows, actual.unbind(), strict=True):
        torch.testing.assert_close(sample, expected)


def test_correct_nested_position_layout_is_unchanged():
    values = torch.arange(4 * 12).reshape(4, 12)
    positions = torch.nested.nested_tensor_from_jagged(values, torch.tensor([0, 5, 12]), jagged_dim=2)
    data = {"position_ids": positions}

    normalize_nested_position_ids(data)

    assert data["position_ids"] is positions


@pytest.mark.parametrize(
    "positions", [torch.arange(12).reshape(2, 6), torch.nested.as_nested_tensor([torch.arange(3)], layout=torch.jagged)]
)
def test_standard_position_ids_are_unchanged(positions):
    data = {"position_ids": positions}
    normalize_nested_position_ids(data)
    assert data["position_ids"] is positions


def test_ambiguous_nested_layout_is_rejected():
    positions = torch.nested.as_nested_tensor([torch.zeros(4, 8), torch.zeros(4, 8)], layout=torch.jagged)
    with pytest.raises(ValueError, match="Cannot identify the ragged axis"):
        normalize_nested_position_ids({"position_ids": positions})


def test_input_token_lengths_disambiguate_square_packed_buffers():
    rows = [torch.arange(32).reshape(4, 8), torch.arange(32, 64).reshape(4, 8)]
    data = TensorDict(
        {
            "position_ids": torch.nested.as_nested_tensor(rows, layout=torch.jagged),
            "input_ids": torch.nested.as_nested_tensor([torch.arange(8), torch.arange(8)], layout=torch.jagged),
        },
        batch_size=2,
    )

    normalize_nested_position_ids(data)

    assert data["position_ids"].offsets().diff().tolist() == [8, 8]
    for expected, sample in zip(rows, data["position_ids"].unbind(), strict=True):
        torch.testing.assert_close(sample, expected)


def test_explicit_lengths_are_rejected_without_dropping_them():
    lengths = torch.tensor([3, 4])
    positions = torch.nested.nested_tensor_from_jagged(
        torch.arange(40).reshape(4, 10), torch.tensor([0, 5, 10]), lengths=lengths, jagged_dim=2
    )
    positions._ragged_idx = 1
    data = {"position_ids": positions}

    with pytest.raises(ValueError, match="must be packed"):
        normalize_nested_position_ids(data)

    assert data["position_ids"] is positions
    torch.testing.assert_close(positions.lengths(), lengths)
