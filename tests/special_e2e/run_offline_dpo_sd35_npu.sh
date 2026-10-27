#!/usr/bin/env bash
# SD3.5 offline DPO e2e on Ascend (NPU port of the doc-referenced GPU manual
# entry run_offline_dpo_sd35.sh).
#
# Offline path: precomputed latents/embeddings parquet -> offline DPO dataset
# -> DirectPreferenceRayTrainer -> FSDP LoRA actor-only update (no rollout,
# no reward server). The GPU entry is single-device by design and its
# micro-batch formula (NUM_PAIRS * 2 per GPU) only divides on one device, so
# this wrapper pins NUM_GPUS=1 and one visible NPU instead of reusing the
# even-count lib_npu_e2e topology; CANN/ATB sourcing, device validation and
# the trainer.device=npu plus Ray visibility runtime args are replicated here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NUM_GPUS=1
export DEVICE_NAME=npu
export ASCEND_RT_VISIBLE_DEVICES="${SD35_NPU_ID:-0}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"

set +u
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

python3 - <<'PY'
import torch
import torch_npu  # noqa: F401

if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("SD3.5 offline DPO NPU smoke requires one visible NPU.")
print(f"NPU offline-DPO smoke: 1 device; torch {torch.__version__}; torch-npu {torch_npu.__version__}")
PY

export MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/stable-diffusion-3-tiny-random}"
export DATA_DIR="${DATA_DIR:-${HOME}/data/dummy_offline_dpo_npu}"
unset TRAIN_FILES VAL_FILES DATA_FILE

NPU_RUNTIME_ARGS=(
    "trainer.device=npu"
    "++ray_kwargs.ray_init.runtime_env.env_vars.ASCEND_VISIBLE_DEVICES=\"${ASCEND_VISIBLE_DEVICES}\""
    '++ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="1"'
    '++ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"'
    '++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE="1"'
)

exec bash "${SCRIPT_DIR}/run_offline_dpo_sd35.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    actor_rollout_ref.model.attn_backend=_native_npu \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    "$@"
