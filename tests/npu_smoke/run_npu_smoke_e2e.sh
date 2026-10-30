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
run_test 18 "Qwen-Image FlowGRPO V1 sync trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_v1_npu.sh
run_test 19 "Qwen-Image-Edit FlowGRPO V1 sync trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_edit_npu.sh
run_test 20 "Qwen-Image online DPO V1 sync trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_online_dpo_qwen_image_npu.sh
run_test 21 "DiffusionNFT V1 sync trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_diffusionnft_qwen_image_npu.sh
run_test 22 "FlowGRPO V1 separate-async trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_v1_separate_async_npu.sh
run_test 23 "Diffusion OPD V1 sync colocated teachers e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    SMOKE=sync bash tests/special_e2e/run_diffusion_teacher_smoke_npu.sh
run_test 24 "Diffusion OPD V1 separate-async standalone teachers e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    SMOKE=async bash tests/special_e2e/run_diffusion_teacher_smoke_npu.sh
run_test 25 "Bagel PickScore LoRA FlowGRPO e2e (v0)" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_bagel_pickscore_npu.sh
run_test 26 "FlowGRPO synchronous separate trainer e2e (v0)" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_separate_npu.sh
run_test 30 "named image previews and H3 offload placeholders" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -q tests/trainer/diffusion/test_artifact_preview_dump_npu.py \
        tests/pipelines/test_minimax_h3_meta_output_npu.py
run_test 34 "MiniMax-H3 T2VA FlowGRPO trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u DATA_DIR -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_minimax_h3_npu.sh
# Qwen3-Omni 8-10 await clean-baseline worker/lifecycle verification;
# MiniCPM-o 27 and Boogu 31 remain blocked. TTS 29 needs packages/assets that
# the current image/workflow does not provision. These are not passing entries.
cleanup_runtime
print_summary
