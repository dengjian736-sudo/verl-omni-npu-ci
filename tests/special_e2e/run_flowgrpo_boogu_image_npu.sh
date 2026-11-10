#!/usr/bin/env bash
# Boogu-Image FlowGRPO e2e on Ascend (NPU port of the GPU manual entry
# run_flowgrpo_boogu_image.sh; MODE=t2i default, MODE=edit exercises the
# reference-latents refiner branch).
#
# Reuses the GPU launcher body (v0 trainer, tiny Boogu built offline from
# precached source assets). Requires the boogu-image package and OpenCV.
# lib_npu_e2e.sh supplies CANN/ATB sourcing, device validation, the NZ
# guards and the trainer.device=npu plus Ray visibility runtime args.
# Manual tier: not registered in the smoke runner, matching the GPU side.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export MODEL_PATH="${BOOGU_MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/Boogu-Image}"
export TOKENIZER_PATH="${BOOGU_TOKENIZER_PATH:-${MODEL_PATH}/processor}"
export DATA_DIR="${BOOGU_DATA_DIR:-${HOME}/data/flowgrpo_boogu_npu/$(date +%Y%m%d_%H%M%S)_$$}"
unset TRAIN_FILES VAL_FILES

exec bash "${SCRIPT_DIR}/run_flowgrpo_boogu_image.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    "$@"
