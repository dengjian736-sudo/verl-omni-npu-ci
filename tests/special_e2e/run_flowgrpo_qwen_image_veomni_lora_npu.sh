#!/usr/bin/env bash
# VeOmni actor + LoRA FlowGRPO e2e on Ascend (NPU port of the GPU manual
# entry run_flowgrpo_qwen_image_veomni_lora.sh).
#
# Reuses the GPU launcher body including its BACKEND=veomni|fsdp2 A/B arms.
# Requires veomni>=0.1.12 installed --no-deps exactly as the GPU smoke
# prepare action does. lib_npu_e2e.sh supplies CANN/ATB sourcing, device
# validation, the NZ guards and the trainer.device=npu plus Ray visibility
# runtime args. Manual tier: not registered in the smoke runner until its
# NPU result is established.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image}"
export DATA_DIR="${DATA_DIR:-${HOME}/data/dummy_diffusion_veomni_npu}"
unset TRAIN_FILES VAL_FILES

exec bash "${SCRIPT_DIR}/run_flowgrpo_qwen_image_veomni_lora.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    "$@"
