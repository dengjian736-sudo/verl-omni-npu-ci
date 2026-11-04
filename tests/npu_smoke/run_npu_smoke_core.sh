#!/usr/bin/env bash

set -euo pipefail

export NPU_SMOKE_GROUP=core
source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"

dispatch_parallel
print_header
run_test 0 "vllm-omni rollout + sleep/wake_up" \
    python3 -m pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py
run_test 3 "diffusers FSDP engine inference + train" \
    python3 -m pytest -s tests/workers/test_diffusers_fsdp_engine_npu.py
run_test 4 "diffusion agent loop request + step execution" --max-npus 4 \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_diffusion_single_turn_npu
run_test 5 "unseeded multi-worker rollout diversity" --max-npus 4 \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_without_seed_npu
run_test 6 "seeded multi-worker rollout reproducibility" --max-npus 4 \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_seed_reproducibility_npu
run_test 7 "standalone vLLM reward engine startup + OCR scoring" --max-npus 4 \
    python3 -m pytest -s tests/reward_loop/test_reward_engine_npu.py::test_standalone_reward_engine_npu
run_test 11 "reward restart determinism and JPEG rule scoring" --max-npus 4 \
    python3 -m pytest -s \
        tests/reward_loop/test_reward_engine_npu.py::test_deterministic_reward_reproducibility_single_replica_npu \
        tests/reward_loop/test_reward_engine_npu.py::test_deterministic_reward_reproducibility_multi_replica_npu \
        tests/reward_loop/test_reward_engine_npu.py::test_rule_reward_npu
run_test 12 "rollout request batching + step-execution contracts" --max-npus 4 \
    python3 -m pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_contracts_npu.py
run_test 17 "OmniAgentLoopWorker image_gen tool agent loop" --max-npus 4 \
    python3 -m pytest -s tests/agent_loop/test_omni_agent_loop_worker_npu.py
cleanup_runtime
print_summary
