#!/usr/bin/env bash
# MiniCPM-o 4.5 Thinker GSPO LoRA e2e on Ascend (ports GPU omni E2E ID 4,
# collocated sync V1 with frozen vpm/apm towers kept in the graph via
# exclude_modules + FSDP2 ignored params).
#
# Reuses the GPU launcher body (tiny random remote-code checkpoint build,
# dummy AVQA data, sdpa attention, LoRA merge sync). lib_npu_e2e.sh supplies
# CANN/ATB sourcing, visible-device validation and the trainer.device=npu
# plus Ray visibility runtime args. Rollout TP stays at 2 on eight NPUs (the
# GPU two-device smoke runs TP=NUM_GPUS=2), so the actor uses all visible
# ranks while rollout forms TP=2 replicas.
#
# The builder downloads the pinned MiniCPM-o remote-code source checkpoint on
# first use; pre-cache it once (HF online) because test runs execute with
# HF_HUB_OFFLINE=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/MiniCPM-o-4_5}"
export DATA_DIR="${DATA_DIR:-${HOME}/data/minicpmo_avqa_npu}"
export TOTAL_TRAIN_STEPS="${TOTAL_TRAIN_STEPS:-2}"

exec bash "${SCRIPT_DIR}/run_gspo_minicpmo_thinker_lora_v1_smoke.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.experiment_name=gspo-minicpmo-thinker-lora-npu-e2e \
    "$@"
