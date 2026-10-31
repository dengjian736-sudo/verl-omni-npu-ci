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
"""CPU coverage of seeded AR sampling and accelerator RNG isolation."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.qwen_image_dual_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithDualLogProb


class _DeviceInput:
    def __init__(self, tensor, device):
        self.tensor = tensor
        self.device = device
        self.shape = tensor.shape

    def to(self, device):
        assert device == self.device
        return self.tensor


@pytest.mark.parametrize("device_type", ["cpu", "xpu"])
@pytest.mark.parametrize("generation_fails", [False, True])
def test_seeded_ar_restores_rng_on_selected_device(monkeypatch, device_type, generation_fails):
    """Seeded generation repeats without changing CPU or the selected accelerator RNG."""
    device = torch.device(device_type, 2) if device_type != "cpu" else torch.device("cpu")
    accelerator_generator = torch.Generator().manual_seed(912)
    saved_devices = []
    restored_devices = []

    def get_accelerator_state(selected_device):
        saved_devices.append(selected_device)
        return accelerator_generator.get_state()

    def set_accelerator_state(state, selected_device):
        restored_devices.append(selected_device)
        accelerator_generator.set_state(state)

    def select_accelerator(selected_device):
        assert selected_device == device
        return nullcontext()

    def reject_cuda(*args, **kwargs):
        raise AssertionError("AR sampling must use the pipeline device RNG")

    monkeypatch.setattr(torch.cuda, "get_rng_state", reject_cuda)
    monkeypatch.setattr(torch.xpu, "get_rng_state", get_accelerator_state)
    monkeypatch.setattr(torch.xpu, "set_rng_state", set_accelerator_state)
    monkeypatch.setattr(torch.xpu, "manual_seed", accelerator_generator.manual_seed)
    monkeypatch.setattr(torch.xpu, "device", select_accelerator)

    def generate(input_ids, attention_mask, max_new_tokens, **kwargs):
        assert attention_mask.shape == input_ids.shape
        generator = accelerator_generator if device_type != "cpu" else None
        response = torch.randint(1, 128, (1, max_new_tokens), generator=generator)
        torch.rand(1)  # Generation can also consume CPU RNG state.
        if generation_fails:
            raise RuntimeError("generation failed")
        return SimpleNamespace(
            sequences=torch.cat((input_ids, response), dim=1),
            scores=tuple(torch.zeros(1, 128) for _ in range(max_new_tokens)),
        )

    pipeline = SimpleNamespace(
        device=device,
        text_encoder=SimpleNamespace(generate=generate),
        tokenizer=SimpleNamespace(pad_token_id=0, batch_decode=lambda ids, **kwargs: [str(ids.tolist())]),
    )
    prompt_ids = _DeviceInput(torch.tensor([[1, 2]]), device)
    attention_mask = _DeviceInput(torch.ones(1, 2, dtype=torch.long), device)
    cpu_state = torch.get_rng_state()
    accelerator_state = accelerator_generator.get_state()

    def sample(seed):
        return QwenImagePipelineWithDualLogProb._get_qwen_text_response(
            pipeline, prompt_ids, attention_mask, seed=seed, max_new_tokens=6
        )

    if generation_fails:
        with pytest.raises(RuntimeError, match="generation failed"):
            sample(42)
    else:
        first = sample(42)
        second = sample(42)
        different = sample(43)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
        assert first[2] == second[2]
        assert not torch.equal(first[0], different[0])

    torch.testing.assert_close(torch.get_rng_state(), cpu_state)
    torch.testing.assert_close(accelerator_generator.get_state(), accelerator_state)
    if device_type != "cpu":
        assert saved_devices == [device] * (1 if generation_fails else 3)
        assert restored_devices == saved_devices
