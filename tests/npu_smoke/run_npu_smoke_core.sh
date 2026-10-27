#!/usr/bin/env bash

set -euo pipefail

export NPU_SMOKE_GROUP=core
source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"

print_header
run_test 0 "vllm-omni rollout + sleep/wake_up" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py
run_test 3 "diffusers FSDP engine inference + train" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    pytest -s tests/workers/test_diffusers_fsdp_engine_npu.py
run_test 4 "diffusion agent loop request + step execution" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_diffusion_single_turn_npu
run_test 5 "unseeded multi-worker rollout diversity" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_without_seed_npu
run_test 6 "seeded multi-worker rollout reproducibility" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_seed_reproducibility_npu
run_test 11 "standalone vLLM reward engine startup + OCR scoring" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_reward_engine_npu.sh
run_test 12 "rollout request batching + step-execution contracts" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_contracts_npu.py
run_test 13 "diffusers FSDP/FSDP2 engine with sequence parallel 2" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    pytest -s tests/workers/test_diffusers_fsdp_engine_sp_npu.py
run_test 14 "diffusers ulysses sp equivalence (torchrun)" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    torchrun --nproc_per_node="${NUM_NPUS}" --local-ranks-filter=0 tests/workers/test_diffusers_ulysses.py
run_test 15 "trainable FSDP2 snapshot restore (two-rank)" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" REQUIRE_TRAINABLE_SNAPSHOT_GPU=1 \
    python3 -m pytest -s tests/workers/test_trainable_snapshot_gpu.py
run_test 16 "composite agent loop AR+DiT single turn (both reward modes)" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_composite_agent_loop_npu.py
run_test 17 "OmniAgentLoopWorker image_gen tool agent loop" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_omni_agent_loop_worker_npu.py
cleanup_runtime
print_summary
