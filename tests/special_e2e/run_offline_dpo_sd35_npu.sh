#!/usr/bin/env bash
# SD3.5 offline DPO e2e on Ascend (NPU port of the doc-referenced GPU manual
# entry run_offline_dpo_sd35.sh).
#
# Offline path: precomputed latents/embeddings parquet -> offline DPO dataset
# -> DirectPreferenceRayTrainer -> FSDP LoRA actor-only update (no rollout,
# no reward server). lib_npu_e2e.sh supplies CANN/ATB sourcing, device
# validation and trainer.device=npu plus Ray visibility runtime args;
# _native_npu attention replaces the GPU product default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/stable-diffusion-3-tiny-random}"
export DATA_DIR="${DATA_DIR:-${HOME}/data/dummy_offline_dpo_npu}"
unset TRAIN_FILES VAL_FILES DATA_FILE

exec bash "${SCRIPT_DIR}/run_offline_dpo_sd35.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    actor_rollout_ref.model.attn_backend=_native_npu \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    "$@"
