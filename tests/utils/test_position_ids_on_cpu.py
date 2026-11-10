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
"""Packed position IDs survive worker minibatch selection without token loss."""

import importlib.util
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

_SPEC = importlib.util.spec_from_file_location(
    "position_ids_under_test", Path(__file__).parents[2] / "verl_omni/utils/position_ids.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
normalize_nested_position_ids = _MODULE.normalize_nested_position_ids


@pytest.mark.parametrize("lengths", [(187, 187), (87, 100), (8, 8), (4, 4), (187,)])
@pytest.mark.parametrize("axes", [3, 4])
@pytest.mark.parametrize("stale_axis", [False, True])
def test_position_ids_preserve_samples_through_worker_minibatches(lengths, axes, stale_axis):
    rows = [torch.arange(axes * length).reshape(axes, length) + index for index, length in enumerate(lengths)]
    positions = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    if stale_axis:
        positions._ragged_idx = 3 - positions._ragged_idx
    data = TensorDict(
        {
            "position_ids": positions,
            "input_ids": torch.nested.as_nested_tensor([torch.arange(n) for n in lengths], layout=torch.jagged),
        },
        batch_size=len(rows),
    )

    normalize_nested_position_ids(data)
    mini_batches = tu.make_iterator(data, mini_batch_size=1, epochs=1, dataloader_kwargs={"shuffle": False})

    for expected, mini_batch in zip(rows, mini_batches, strict=True):
        # Pinned verl reconstructs nested tensors during selection; the worker
        # normalizes again before its engine consumes each minibatch.
        normalize_nested_position_ids(mini_batch)
        actual = mini_batch["position_ids"]
        assert actual.shape[1] == axes
        assert actual.offsets().tolist() == [0, expected.shape[1]]
        assert actual.dtype == torch.long
        torch.testing.assert_close(actual.unbind()[0], expected, rtol=0, atol=0)


def test_correct_packed_positions_reuse_the_values_buffer():
    values = torch.arange(48).reshape(4, 12)
    offsets = torch.tensor([0, 5, 12])
    positions = torch.nested.nested_tensor_from_jagged(values, offsets, jagged_dim=2)
    data = {
        "position_ids": positions,
        "input_ids": torch.nested.nested_tensor_from_jagged(torch.arange(12), offsets),
    }

    normalize_nested_position_ids(data)

    assert data["position_ids"].values().data_ptr() == values.data_ptr()
    torch.testing.assert_close(data["position_ids"].values(), values, rtol=0, atol=0)


@pytest.mark.parametrize(
    "positions", [torch.arange(12).reshape(2, 6), torch.nested.as_nested_tensor([torch.arange(3)], layout=torch.jagged)]
)
def test_standard_position_ids_are_unchanged(positions):
    data = {"position_ids": positions}
    normalize_nested_position_ids(data)
    assert data["position_ids"] is positions


def test_mismatched_token_lengths_are_rejected():
    positions = torch.nested.as_nested_tensor([torch.zeros(4, 7), torch.zeros(4, 7)], layout=torch.jagged)
    data = {
        "position_ids": positions,
        "input_ids": torch.nested.as_nested_tensor([torch.arange(6), torch.arange(7)], layout=torch.jagged),
    }
    with pytest.raises(ValueError, match="do not match"):
        normalize_nested_position_ids(data)
    assert data["position_ids"] is positions


def test_explicit_lengths_are_rejected_without_dropping_tokens():
    offsets = torch.tensor([0, 5, 10])
    positions = torch.nested.nested_tensor_from_jagged(
        torch.arange(40).reshape(4, 10), offsets, lengths=torch.tensor([3, 4]), jagged_dim=2
    )
    data = {
        "position_ids": positions,
        "input_ids": torch.nested.nested_tensor_from_jagged(torch.arange(10), offsets),
    }
    with pytest.raises(ValueError, match="must be packed"):
        normalize_nested_position_ids(data)
    assert data["position_ids"] is positions
