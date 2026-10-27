#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO with VeOmni actor on Ascend (NPU port of the
# doc-referenced GPU manual entry run_gspo_qwen3_omni_thinker_veomni_smoke.sh).
#
# Standalone (not a wrapper): the GPU launcher rebuilds its tiny checkpoint
# with the 32-wide head default, while the Ascend inference attention kernel
# needs the 128-wide fixture (same choice as the Qwen3-Omni GSPO smoke).
# Keeps the GPU launcher's two-step V1 GSPO contract, VeOmni offload=false
# arms and full-weight-sync coverage. Requires veomni>=0.1.12 installed
# --no-deps as the GPU smoke prepare action does. Manual tier: not
# registered in the smoke runner until its NPU result is established.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/lib_npu_e2e.sh"

export MODEL_PATH="${QWEN3_OMNI_VEOMNI_MODEL_PATH:-${HOME}/.cache/verl-omni-npu-smoke/Qwen3-Omni-VeOmni}"
DATA_DIR="${QWEN3_OMNI_DATA_DIR:-${HOME}/data/math_veomni_npu}"
export NUM_GPUS="${NUM_NPUS}"
export NNODES=1 ACTOR_EP=1 ROLLOUT_TP=2
unset TRAIN_FILES VAL_FILES

python3 "${SCRIPT_DIR}/build_qwen3_omni_tiny_random.py" \
    --output-dir "${MODEL_PATH}" --force --head-dim 128
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 "${SCRIPT_DIR}/create_dummy_math_data.py" --local_save_dir "${DATA_DIR}"
fi

TRAIN_FILE="${DATA_DIR}/train.parquet" VAL_FILE="${DATA_DIR}/test.parquet" \
exec bash "${REPO_ROOT}/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni.sh" \
    "${NPU_RUNTIME_ARGS[@]}" \
    data.dataloader_num_workers=0 \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    data.val_max_samples=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.veomni.param_offload=false \
    actor_rollout_ref.actor.veomni.optimizer_offload=false \
    actor_rollout_ref.ref.veomni.param_offload=false \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=1800 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.init_timeout=1800 \
    reward.custom_reward_function.path=null \
    trainer.val_before_train=false \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS:-2}" \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=qwen3-omni-thinker-veomni-npu-smoke \
    "$@"
