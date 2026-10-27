#!/usr/bin/env bash
# Boogu-Image DiffusionNFT trainer e2e on Ascend (ports GPU diffusion E2E
# ID 10).
#
# Reuses the GPU launcher body (tiny Boogu build, MODE=t2i default, native
# attention already pinned there). The launcher's preflight accepts torch_npu
# when the CUDA build is absent. Requires the boogu-image package and OpenCV
# in the environment. lib_npu_e2e.sh supplies CANN/ATB sourcing, device
# validation and the trainer.device=npu plus Ray visibility runtime args.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export NUM_GPUS="${NUM_NPUS}"
export BOOGU_REQUIRED=1
export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/Boogu-Image}"
export DATA_DIR="${DATA_DIR:-${HOME}/data/dummy_boogu_npu}"
unset TRAIN_FILES VAL_FILES

exec bash "${SCRIPT_DIR}/run_diffusionnft_boogu_image.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    trainer.experiment_name=diffusionnft-boogu-image-npu-e2e \
    "$@"
