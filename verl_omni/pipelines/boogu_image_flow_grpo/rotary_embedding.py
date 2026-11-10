# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Boogu Team.
# Original work: Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.
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
"""Portable lookup for the canonical Boogu rotary embedding.

Adapted from Boogu-Image's rope.py at commit ``25f8f888``:
https://github.com/boogu-project/Boogu-Image/blob/25f8f888/boogu/models/transformers/rope.py.
"""

import torch
from boogu.models.transformers.rope import BooguImageDoubleStreamRotaryPosEmbed


class BooguImagePortableRotaryPosEmbed(BooguImageDoubleStreamRotaryPosEmbed):
    """Gather the real and imaginary components without requiring complex gather."""

    def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
        if ids.device.type == "mps":
            return super()._get_freqs_cis(freqs_cis, ids)

        result = []
        for axis in range(len(self.axes_dim)):
            freqs = torch.view_as_real(freqs_cis[axis]).to(ids.device)
            index = ids[:, :, axis : axis + 1, None].to(torch.int64)
            index = index.expand(-1, -1, freqs.shape[-2], 2)
            selected = torch.gather(freqs.unsqueeze(0).expand(index.shape[0], -1, -1, -1), dim=1, index=index)
            result.append(selected)
        return torch.view_as_complex(torch.cat(result, dim=-2))
