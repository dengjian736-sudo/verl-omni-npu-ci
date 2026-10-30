#!/usr/bin/env bash
# Run the existing Qwen3-Omni GPU smoke workloads on Ascend.
set -euo pipefail

MODE=${1:-}
case "${MODE}" in
    gspo|offline-dpo|separate-async) shift ;;
    -h|--help)
        echo "Usage: $0 {gspo|offline-dpo|separate-async} [Hydra overrides...]"
        exit 0
        ;;
    *)
        echo "Expected gspo, offline-dpo, or separate-async." >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export VLLM_ASCEND_ENABLE_NZ=0
NPU_RUNTIME_ARGS+=(
    '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ="0"'
    data.dataloader_num_workers=0
)

unset TOKENIZER_PATH TRAIN_FILES VAL_FILES
DATA_DIR="${QWEN3_OMNI_DATA_DIR:-${REPO_ROOT}/logs/npu_smoke/qwen3_omni/$(date +%Y%m%d_%H%M%S)_$$}/${MODE}"
MODEL_PATH="${QWEN3_OMNI_MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/Qwen3-Omni}"
if [[ "${MODE}" == offline-dpo ]]; then
    MODEL_PATH="${QWEN3_OMNI_MULTIMODAL_MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/Qwen3-Omni-Multimodal}"
fi
export DATA_DIR MODEL_PATH

# The GPU launchers build tiny checkpoints; never overwrite a full model or unrelated cache.
python3 - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
if path.is_symlink():
    raise SystemExit(f"Tiny-model output must not be a symbolic link: {path}")
if path.exists() and any(path.iterdir()):
    config_path = path / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Use an empty tiny-model output directory; config.json is missing: {path}")
    config = json.loads(config_path.read_text())
    text_config = config.get("thinker_config", {}).get("text_config", {})
    if (
        config.get("model_type") != "qwen3_omni_moe"
        or text_config.get("num_hidden_layers") != 2
        or text_config.get("hidden_size") != 128
        or any(item.is_symlink() for item in path.iterdir())
    ):
        raise SystemExit(f"Refusing to rebuild a directory that is not the local tiny Qwen3-Omni model: {path}")
PY

default_steps=2
[[ "${MODE}" != separate-async ]] || default_steps=3
TOTAL_TRAIN_STEPS=${QWEN3_OMNI_TOTAL_TRAIN_STEPS:-${default_steps}}
if [[ ! "${TOTAL_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]] || (( TOTAL_TRAIN_STEPS < default_steps )); then
    echo "${MODE} requires at least ${default_steps} training steps." >&2
    exit 2
fi

if [[ "${MODE}" == offline-dpo ]]; then
    export NUM_GPUS="${NUM_NPUS}"
    export TRAIN_BATCH_SIZE=$(( (4 + NUM_NPUS - 1) / NUM_NPUS * NUM_NPUS ))
    export PPO_MINI_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
    export PPO_MICRO_BATCH_SIZE_PER_GPU=1
    export VAL_BATCH_SIZE="${NUM_NPUS}"
    export TRAIN_SIZE=$(( TRAIN_BATCH_SIZE * TOTAL_TRAIN_STEPS ))
    export VAL_SIZE="${VAL_BATCH_SIZE}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAIN_STEPS}"
    export CHECKPOINT_DIR="${QWEN3_OMNI_CHECKPOINT_DIR:-${DATA_DIR}/checkpoints}"
    export ATTN_IMPLEMENTATION=sdpa
    python3 "${SCRIPT_DIR}/create_dummy_omni_preference_dpo_data.py" \
        --local_save_dir "${DATA_DIR}" --train_size "${TRAIN_SIZE}" --val_size "${VAL_SIZE}"
    exec bash "${SCRIPT_DIR}/run_qwen3_omni_multimodal_offline_mllm_dpo_lora_smoke.sh" \
        "${NPU_RUNTIME_ARGS[@]}" \
        trainer.experiment_name=qwen3-omni-multimodal-dpo-npu-smoke \
        "$@"
fi

NUM_TRAIN_NPUS="${NUM_NPUS}"
ROLLOUT_TP=2
export TINY_HEAD_DIM=128
if [[ "${MODE}" == separate-async ]]; then
    NUM_TRAIN_NPUS=$(( NUM_NPUS / 2 ))
    NUM_ROLLOUT_NPUS=$(( NUM_NPUS - NUM_TRAIN_NPUS ))
    if (( NUM_TRAIN_NPUS % 2 != 0 || NUM_ROLLOUT_NPUS % 2 != 0 )); then
        ROLLOUT_TP=1
    fi
fi
mini_batch_size=$(( (4 + NUM_TRAIN_NPUS - 1) / NUM_TRAIN_NPUS * NUM_TRAIN_NPUS ))
train_batch_size="${mini_batch_size}"
if [[ "${MODE}" == separate-async ]]; then
    train_batch_size=$(( 2 * mini_batch_size ))
fi
val_size=$(( (4 + NUM_NPUS - 1) / NUM_NPUS * NUM_NPUS ))
python3 "${SCRIPT_DIR}/create_dummy_math_data.py" \
    --local_save_dir "${DATA_DIR}" \
    --train_size "$(( train_batch_size * TOTAL_TRAIN_STEPS * 2 ))" --val_size "${val_size}"

NPU_RUNTIME_ARGS+=(
    "data.train_batch_size=${train_batch_size}"
    "data.val_max_samples=${val_size}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.rollout.enforce_eager=True
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0
    "trainer.experiment_name=qwen3-omni-${MODE}-npu-smoke"
)
# Math-only GSPO never consumes the tiny model's GPU-shaped image/video/audio towers.
NPU_RUNTIME_ARGS+=('++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt={image:0,video:0,audio:0}')

export TOTAL_TRAIN_STEPS
if [[ "${MODE}" == gspo ]]; then
    export NUM_GPUS="${NUM_NPUS}"
    exec bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_thinker_lora_v1_smoke.sh" \
        "${NPU_RUNTIME_ARGS[@]}" "$@"
else
    export NUM_TRAIN_GPUS="${NUM_TRAIN_NPUS}" NUM_ROLLOUT_GPUS="${NUM_ROLLOUT_NPUS}"
    exec bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_thinker_lora_v1_separate_async_smoke.sh" \
        "${NPU_RUNTIME_ARGS[@]}" "$@"
fi
