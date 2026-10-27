#!/usr/bin/env bash
# FlowGRPO V1 separate_async trainer e2e on Ascend (ports GPU diffusion E2E
# ID 4).
#
# Reuses the GPU launcher body with the HCCL checkpoint engine (verl's
# Ascend counterpart of the cupy-based NCCL engine); at eight NPUs the
# default split is four actor ranks plus four standalone TP=1 rollout
# replicas. lib_npu_e2e.sh supplies CANN/ATB sourcing, visible-device
# validation and the trainer.device=npu plus Ray visibility runtime args.
# The GPU launcher selects native/TORCH_SDPA attention automatically because
# fa_available() is false on the pinned Ascend stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export CKPT_BACKEND="${CKPT_BACKEND:-hccl}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image}"
export TOKENIZER_PATH="${MODEL_PATH}/tokenizer"
export DATA_DIR="${DATA_DIR:-${HOME}/data/dummy_diffusion_async_npu}"
unset TRAIN_FILES VAL_FILES

exec bash "${SCRIPT_DIR}/run_flowgrpo_qwen_image_v1_separate_async.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    trainer.experiment_name=flowgrpo-diffusion-v1-separate-async-npu-e2e \
    "$@"
