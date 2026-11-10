#!/usr/bin/env bash
# Provision optional model packages and source assets before offline E2E runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
case "${1:-}" in
    '')
        python3 -m pip install --no-deps -r tests/npu_smoke/requirements_model_smoke.txt
        python3 -m pip install --no-deps --force-reinstall \
            "qwen-tts @ https://github.com/QwenLM/Qwen3-TTS/archive/$(cat .github/qwen_tts_pin.txt).tar.gz" \
            "boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git@$(cat .github/boogu_image_pin.txt)"
        if ! command -v sox >/dev/null; then
            apt-get update
            apt-get install -y --no-install-recommends sox
        fi
        ;;
    --check-only) ;;
    *) echo "Usage: $0 [--check-only]" >&2; exit 2 ;;
esac

command -v sox >/dev/null
python3 - <<'PY'
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration  # noqa: F401
from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel  # noqa: F401
import av, cv2, librosa, onnxruntime, qwen_omni_utils, qwen_vl_utils, soundfile, sox, torchvision  # noqa: F401
PY

if [[ "${1:-}" != --check-only ]]; then
    python3 - <<'PY'
from huggingface_hub import snapshot_download
from tests.special_e2e.build_boogu_image_tiny_random import (
    DEFAULT_SOURCE_MODEL, DEFAULT_SOURCE_REVISION,
)

snapshot_download(DEFAULT_SOURCE_MODEL, revision=DEFAULT_SOURCE_REVISION,
                  allow_patterns=['processor/*', 'scheduler/*'], endpoint='https://huggingface.co')
# Match the pinned source used by run_qwen3_tts_grpo_smoke.sh.
snapshot_download('optimum-intel-internal-testing/tiny-random-qwen3-tts',
                  revision='6374d605b31381cac6f9577f5e742af2f76ba79c')
PY
fi
