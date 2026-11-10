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
"""Compare portable Boogu RoPE lookup and gradients with the canonical implementation."""

import pytest
import torch

pytest.importorskip("boogu")
from boogu.models.embeddings import apply_rotary_emb
from boogu.models.transformers.rope import BooguImageDoubleStreamRotaryPosEmbed

from verl_omni.pipelines.boogu_image_flow_grpo.rotary_embedding import BooguImagePortableRotaryPosEmbed


@pytest.mark.parametrize("frequency_dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
def test_real_gather_matches_complex_lookup_and_training_gradients(monkeypatch, frequency_dtype, activation_dtype):
    """Repeated positions and noncontiguous tables preserve exact values and gradients."""
    generator = torch.Generator().manual_seed(123)
    axes_dim = (4, 6, 2)
    axes_lens = (7, 11, 13)
    original = BooguImageDoubleStreamRotaryPosEmbed(10000, axes_dim, axes_lens)
    portable = BooguImagePortableRotaryPosEmbed(10000, axes_dim, axes_lens)
    tables = [
        torch.randn(length, dim, dtype=frequency_dtype, generator=generator)[:, ::2]
        for dim, length in zip(axes_dim, axes_lens, strict=True)
    ]
    reference_tables = [table.detach().requires_grad_() for table in tables]
    portable_tables = [table.detach().requires_grad_() for table in tables]
    positions = torch.tensor(
        [
            [[0, 0, 0], [6, 10, 12], [1, 3, 5], [1, 3, 5], [0, 3, 12]],
            [[6, 0, 12], [0, 10, 0], [1, 3, 5], [6, 10, 12], [6, 10, 12]],
        ],
        dtype=torch.int32,
    )
    expected = original._get_freqs_cis(reference_tables, positions)
    gather = torch.gather
    gathered_dtypes = []

    def real_gather(tensor, *args, **kwargs):
        assert not tensor.is_complex(), "complex gather is unavailable on this backend"
        gathered_dtypes.append(tensor.dtype)
        return gather(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "gather", real_gather)
    with pytest.raises(AssertionError, match="complex gather"):
        original._get_freqs_cis(reference_tables, positions)
    actual = portable._get_freqs_cis(portable_tables, positions)
    assert actual.dtype == frequency_dtype
    assert actual.device == positions.device
    assert gathered_dtypes == [torch.float32 if frequency_dtype == torch.complex64 else torch.float64] * 3
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    activations = torch.randn(2, 5, 3, sum(axes_dim), dtype=activation_dtype, generator=generator)
    reference_input = activations.detach().requires_grad_()
    portable_input = activations.detach().requires_grad_()
    reference_output = apply_rotary_emb(reference_input, expected, use_real=False)
    portable_output = apply_rotary_emb(portable_input, actual, use_real=False)
    assert portable_output.dtype == activation_dtype
    torch.testing.assert_close(portable_output, reference_output, rtol=0, atol=0)
    weights = torch.randn(reference_output.shape, generator=generator)
    reference_gradients = torch.autograd.grad((reference_output * weights).sum(), [reference_input, *reference_tables])
    portable_gradients = torch.autograd.grad((portable_output * weights).sum(), [portable_input, *portable_tables])
    for actual_gradient, expected_gradient in zip(portable_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("with_reference", [False, True])
def test_inherited_text_and_image_position_layout_matches_upstream(with_reference):
    """Keep T2I/Edit position construction and stream slicing in the upstream forward."""
    axes_dim = (4, 6, 2)
    axes_lens = (32, 32, 32)
    original = BooguImageDoubleStreamRotaryPosEmbed(10000, axes_dim, axes_lens)
    portable = BooguImagePortableRotaryPosEmbed(10000, axes_dim, axes_lens)
    freqs_cis = original.get_freqs_cis(axes_dim, axes_lens, theta=10000)
    kwargs = {
        "freqs_cis": freqs_cis,
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "l_effective_ref_img_len": [[2], []] if with_reference else [[], []],
        "l_effective_img_len": [4, 2],
        "ref_img_sizes": [[(4, 2)], None] if with_reference else [None, None],
        "img_sizes": [(4, 4), (2, 4)],
        "device": torch.device("cpu"),
    }
    assert BooguImagePortableRotaryPosEmbed.forward is BooguImageDoubleStreamRotaryPosEmbed.forward
    assert portable.state_dict() == original.state_dict() == {}
    for actual, expected in zip(portable(**kwargs), original(**kwargs), strict=True):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected
