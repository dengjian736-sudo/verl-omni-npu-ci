#!/usr/bin/env bash
# Exercise the reward server with a cold, test-owned architecture-inspection cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export DEVICE_NAME=npu
export NUM_NPUS="${NUM_NPUS:-8}"
if [[ ! "${NUM_NPUS}" =~ ^[1-9][0-9]*$ ]] || (( NUM_NPUS < 2 || NUM_NPUS % 2 )); then
    echo "The reward smoke requires an even NUM_NPUS of at least two." >&2
    exit 2
fi
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 "$((NUM_NPUS - 1))")}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ASCEND_ENABLE_NZ=0
export NPU_REWARD_MODEL_PATH="${NPU_REWARD_MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/Qwen2.5-VL-Reward-NPU}"
LOG_ROOT="${LOG_DIR:-${REPO_ROOT}/logs/npu_smoke/reward_engine}"
mkdir -p "${LOG_ROOT}"
export VLLM_CACHE_ROOT
VLLM_CACHE_ROOT=$(mktemp -d "${LOG_ROOT}/reward-registry.XXXXXX")
echo "Reward model-inspection cache: ${VLLM_CACHE_ROOT}"
python3 tests/special_e2e/build_qwen25_vl_reward_tiny_random.py --output-dir "${NPU_REWARD_MODEL_PATH}"
exec python3 -m pytest -v -s tests/reward_loop/test_reward_engine_npu.py
