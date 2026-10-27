#!/usr/bin/env bash

set -euo pipefail

export NPU_SMOKE_GROUP=e2e
source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"

print_header
run_test 1 "FlowGRPO trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_npu.sh
run_test 2 "DanceGRPO Wan2.2 trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_dancegrpo_wan22_npu.sh
run_test 7 "MiniMax-H3 T2VA FlowGRPO trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_minimax_h3_npu.sh
run_test 8 "Qwen3-Omni Thinker GSPO LoRA trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_qwen3_omni_npu.sh gspo
run_test 9 "Qwen3-Omni multimodal offline DPO trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_qwen3_omni_npu.sh offline-dpo
run_test 10 "Qwen3-Omni Thinker separate-async GSPO trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_qwen3_omni_npu.sh separate-async
run_test 18 "Qwen-Image FlowGRPO V1 sync trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_v1_npu.sh
run_test 19 "Qwen-Image-Edit FlowGRPO V1 sync trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_edit_npu.sh
run_test 20 "Qwen-Image online DPO V1 sync trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_online_dpo_qwen_image_npu.sh
run_test 21 "DiffusionNFT V1 sync trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_diffusionnft_qwen_image_npu.sh
run_test 22 "FlowGRPO V1 separate_async trainer e2e (HCCL ckpt)" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_v1_separate_async_npu.sh
run_test 23 "Diffusion OPD V1 sync colocated teachers e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" SMOKE=sync \
    bash tests/special_e2e/run_diffusion_teacher_smoke_npu.sh
run_test 24 "Diffusion OPD V1 separate_async standalone teachers e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" SMOKE=async \
    bash tests/special_e2e/run_diffusion_teacher_smoke_npu.sh
run_test 25 "Bagel PickScore LoRA FlowGRPO e2e (v0 trainer)" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_bagel_pickscore_npu.sh
run_test 26 "FlowGRPO synchronous separate trainer e2e (v0, HCCL ckpt)" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_separate_npu.sh
run_test 27 "MiniCPM-o 4.5 Thinker GSPO LoRA e2e (collocated sync, V1)" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_gspo_minicpmo_thinker_lora_v1_npu.sh
cleanup_runtime
print_summary
