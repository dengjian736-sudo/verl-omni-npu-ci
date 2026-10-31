# Deferred NPU model and worker cases

Last updated: 10/30/2026

These changes extend the GPU-to-NPU migration with functional fixes. They are
separate from the validated default smoke suite in [the NPU guide](npu_smoke_tests.md).
The commands below require a dedicated container, eight available NPUs, the
repository's pinned inference stack, and local model assets. Nonzero exits,
including native teardown aborts, remain failures.

## Functional changes

| Area | Failure | Change |
| --- | --- | --- |
| FSDP1 LoRA export | PEFT derives prefixes containing nested FSDP wrappers, but FSDP `state_dict()` removes them, dropping nested default-adapter weights. | Use the full-summon, named-parameter collector for every non-layered FSDP1 adapter, preserving saved-module buffers and their local names. Keep the layered and FSDP2 paths. |
| Composite AR sampling | `fork_rng` defaults to CUDA even when the pipeline device is NPU. | Pass the actual device type to PyTorch's RNG context. |
| Packed multi-axis positions | Equal-length samples can be packed along the axes dimension; changing private ragged metadata does not repair the shape used for minibatch selection. | Rebuild the jagged tensor through the public API at train/infer worker boundaries, preserving token offsets and all integer positions. Reject inconsistent batches. |
| Rollout wake-up | A weights-only wake resumes admission while KV cache remains asleep. | Resume generation only when the pinned engine reports that no sleeping tags remain. |
| Boogu actor RoPE | Ascend rejects `torch.gather` on complex frequency tables. | Gather real/imaginary components at the original precision and reconstruct the complex view. A small local RoPE subclass inherits upstream position construction and serves both FlowGRPO and NFT adapters. |

The shared worker changes contain no model-name dispatch. Boogu's canonical
model class and installed dependencies are not patched in place. CPU regression
tests cover exact LoRA keys/values, position-preserving minibatch selection,
partial wake admission, RNG restoration, and Boogu output/gradient equivalence.
They do not establish NPU training success.

The Qwen3-Omni GSPO launchers select eager entropy computation through the
existing `fsdp_config.use_torch_compile=False` option. BF16, FSDP2, training
steps and weight synchronization remain exercised; compiled-entropy coverage
is not claimed.

## Preparation and acceptance

Keep the installed torch, torch-npu, vLLM, vLLM-Ascend, vLLM-Omni and verl pins
fixed while diagnosing these cases. Use a separate environment for media extras.
Full Qwen3-Omni needs `qwen-omni-utils==0.0.9`, compatible torchvision/AV,
librosa and soundfile. Boogu and Qwen3-TTS use the source revisions in
`.github/boogu_image_pin.txt` and `.github/qwen_tts_pin.txt`, installed without
replacing the inference stack. Pre-cache source model/tokenizer assets before
offline execution. A dependency import check is not an execution test.
The current image and optional packages still have declared dependency
conflicts, including Boogu's diffusers bound and Qwen3-TTS's Transformers
minimum. Record `pip check` and runtime results; these launchers do not certify
a fully resolved package environment.

Use `BOOGU_MODEL_PATH` / `BOOGU_DATA_DIR` and `MINICPMO_MODEL_PATH` /
`MINICPMO_DATA_DIR` for those launchers; unrelated ambient model paths are ignored.

With the common NPU environment and jemalloc setup from the NPU guide:

```bash
export NUM_NPUS=8 NUM_GPUS=8 DEVICE_NAME=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_NPU_SOCKET_PORT_RANGE=auto

MODEL_PATH="$QWEN_IMAGE_MODEL_PATH" python3 -m pytest -v -s \
  tests/workers/test_diffusers_fsdp_lora_adapter.py
MODEL_PATH="$QWEN_IMAGE_MODEL_PATH" python3 -m pytest -v -s \
  tests/agent_loop/test_composite_agent_loop_npu.py

bash tests/special_e2e/run_qwen3_omni_npu.sh gspo
bash tests/special_e2e/run_qwen3_omni_npu.sh offline-dpo
bash tests/special_e2e/run_qwen3_omni_npu.sh separate-async

MODE=t2i bash tests/special_e2e/run_diffusionnft_boogu_image_npu.sh
MODE=t2i bash tests/special_e2e/run_flowgrpo_boogu_image_npu.sh
```

Run entries sequentially on one server and release their Ray/engine processes
between entries. Preserve each process exit, log-writer exit, expected pytest
counts or training steps, and post-run device/process checks. The existing
default Core/E2E workflow remains split across independent eight-NPU jobs;
these candidate cases are not added to that workflow before hardware validation
and dependency provisioning are established.

MiniCPM-o and Qwen3-TTS have manual NPU launchers with batch sizes divisible by
the actor data-parallel size. Their inference operators still require hardware
validation; a launcher alone is not a functional fix. MiniCPM-o historically
failed in the pinned inference runner's event synchronization (107020).
VeOmni NPU engine registration overlaps existing upstream PRs #727 and #812,
so this branch does not duplicate those engine changes.

Hardware results for the functional changes above are pending. The earlier
default-suite result does not certify these additional cases or their dependency
environment.
