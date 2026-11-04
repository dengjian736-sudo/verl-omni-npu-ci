#!/usr/bin/env bash

set -euo pipefail

export NPU_SMOKE_GROUP=e2e
source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"

dispatch_parallel
print_header
run_test 1 "FlowGRPO trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        DATA_DIR="${LOG_DIR}/test1_data" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_npu.sh
run_test 2 "DanceGRPO Wan2.2 trainer e2e" --max-npus 4 \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        DATA_DIR="${LOG_DIR}/test2_data" \
    bash tests/special_e2e/run_dancegrpo_wan22_npu.sh
cleanup_runtime
print_summary
