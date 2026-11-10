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
"""Exercise packed positions through the production training-worker methods."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.workers.engine_workers import TrainingWorker


@pytest.mark.parametrize("method", ["train_mini_batch", "train_batch", "infer_batch"])
@pytest.mark.parametrize("packing", ["automatic", "upstream"])
def test_training_worker_preserves_multi_axis_positions(monkeypatch, method, packing):
    rows = [torch.arange(28).reshape(4, 7) + index * 100 for index in range(2)]
    positions = (
        torch.nested.as_nested_tensor(rows, layout=torch.jagged)
        if packing == "automatic"
        else tu.nested_tensor_from_tensor_list(rows)
    )
    data = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.arange(7), torch.arange(7)], layout=torch.jagged),
            "position_ids": positions,
        },
        batch_size=2,
    )
    tu.assign_non_tensor(data, mini_batch_size=1)
    seen = []

    def forward(batch, *, loss_function):
        assert loss_function is worker.loss_fn
        assert batch["position_ids"].shape[1] == 4
        seen.extend(batch["position_ids"].unbind())
        return {}

    worker = object.__new__(TrainingWorker)
    worker.model_config = {"use_remove_padding": True}
    worker.engine_config = SimpleNamespace(
        forward_only=False,
        use_dynamic_bsz=False,
        max_token_len_per_gpu=16,
        micro_batch_size_per_gpu=1,
        infer_max_token_len_per_gpu=16,
        infer_micro_batch_size_per_gpu=1,
        use_fused_kernels=False,
    )
    worker.flops_counter = None
    worker.loss_fn = object()
    worker.engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        eval_mode=lambda **kwargs: nullcontext(),
        train_batch=forward,
        infer_batch=forward,
        lr_scheduler_step=lambda: 1e-6,
        is_mp_src_rank_with_outputs=lambda: False,
        get_data_parallel_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_group=lambda: None,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)

    def all_gather_object(output, value, group):
        output[0] = value

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    getattr(worker, method)(data)

    assert len(seen) == len(rows)
    for actual, expected in zip(seen, rows, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
