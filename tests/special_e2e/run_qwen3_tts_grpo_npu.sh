#!/usr/bin/env bash
# Qwen3-TTS Talker full-parameter GRPO e2e on Ascend (ports GPU omni E2E
# ID 3).
#
# Reuses the GPU smoke launcher body, which rebuilds the tiny 16-codebook
# checkpoint from the pinned HF source model, creates dummy GRPO data and
# validates that training reaches exactly global_step 2. Batch sizes are
# rounded to the actor data-parallel size; Ascend selects its attention backend.
#
# Prerequisites in the environment (same installs as the GPU smoke prepare
# action): the pinned qwen-tts archive with --no-deps, plus sox and the
# audio extras. The pinned tiny-random-qwen3-tts source revision must be
# pre-cached in the HF cache because runs execute with HF_HUB_OFFLINE=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export WORK_DIR="${WORK_DIR:-${HOME}/data/qwen3_tts_grpo_npu/$(date +%Y%m%d_%H%M%S)_$$}"
unset MODEL_PATH TRAIN_FILES VAL_FILES

mini_batch_size=$(( (4 + NUM_NPUS - 1) / NUM_NPUS * NUM_NPUS ))

exec bash "${SCRIPT_DIR}/run_qwen3_tts_grpo_smoke.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    "data.train_batch_size=${mini_batch_size}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}" \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.attention_config.backend=null' \
    "$@"
