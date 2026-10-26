# NPU Smoke Tests

Last updated: 10/26/2026.

This guide explains how to add CI test cases for Huawei Ascend devices in verl-omni.

The workflow entry point is .github/workflows/npu_smoke.yml. The suite is split
into these runners:

- tests/npu_smoke/run_npu_smoke_core.sh runs rollout, worker, and engine checks.
- tests/npu_smoke/run_npu_smoke_e2e.sh runs end-to-end trainer checks.
- tests/npu_smoke/run_npu_smoke_tests.sh runs both groups.
- tests/npu_smoke/lib_npu_smoke.sh contains shared cleanup, logging, and summaries.

## Test Cases

| ID | Group | Name | Test entry point | Default NPUs |
|---|---|---|---|---:|
| 0 | core | vLLM-Omni rollout + sleep/wake-up | tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py | 8 |
| 1 | e2e | Qwen-Image FlowGRPO trainer | tests/special_e2e/run_flowgrpo_qwen_image_npu.sh | 8 |
| 2 | e2e | Wan2.2 DanceGRPO trainer | tests/special_e2e/run_dancegrpo_wan22_npu.sh | 8 |
| 3 | core | Diffusers FSDP engine inference and training | tests/workers/test_diffusers_fsdp_engine_npu.py | 8 |
| 7 | e2e | MiniMax-H3 T2VA FlowGRPO trainer | tests/special_e2e/run_flowgrpo_minimax_h3_npu.sh | 8 |
| 8 | e2e | Qwen3-Omni Thinker GSPO LoRA trainer | tests/special_e2e/run_qwen3_omni_npu.sh gspo | 8 |
| 9 | e2e | Qwen3-Omni multimodal offline DPO trainer | tests/special_e2e/run_qwen3_omni_npu.sh offline-dpo | 8 |
| 10 | e2e | Qwen3-Omni Thinker separate-async GSPO trainer | tests/special_e2e/run_qwen3_omni_npu.sh separate-async | 8 |

Test 3 ports the `fsdp` case of `tests/workers/test_diffusers_fsdp_engine.py`
(GPU core ID 2). It reuses the GPU test's synthetic trajectories and engine
inference/training exercise, with `_native_npu` attention and sequence parallel
size 1. It requires at least two visible NPUs and a local tiny Qwen-Image model.
FSDP2 and sequence-parallel execution are not covered by this NPU case.
The test checks finite inference log probabilities, training loss, and gradient
norm; it does not establish GPU/NPU numerical parity or model convergence.

Run only the migrated case from the repository root after sourcing CANN/ATB:

```bash
DEVICE_NAME=npu NUM_NPUS=2 \
MODEL_PATH="${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image" \
python3 -m pytest -v -s tests/workers/test_diffusers_fsdp_engine_npu.py
```

The same case runs through `run_npu_smoke_core.sh` and the combined
`run_npu_smoke_tests.sh` entry point. Missing hardware or model files fail
explicitly rather than producing an all-skipped smoke success.

### MiniMax-H3 and Qwen3-Omni model smoke

Test 7 reuses GPU diffusion E2E ID 7: tiny MiniMax-H3 text-to-video-and-audio
FlowGRPO, one V1 sync training step, FSDP2 with actor sequence parallel size 2
and rollout tensor parallel size 2. Its offline builder generates deterministic
random weights with the pinned GPU fixture's geometry and reuses the existing
tiny VAE stubs and checkpoint conversion. The generated weights differ from
the downloaded GPU fixture; this tests model execution, not output quality or
GPU/NPU numerical parity. FL2VA, Ref2VA, and standalone Ulysses equivalence
checks are outside this case.

Tests 8–10 reuse GPU omni E2E IDs 0–2 respectively:

- GSPO runs two steps with LoRA, rollout generation, and validation.
- Offline DPO runs two steps on synthetic image/video/audio preference data
  and checks the saved actor checkpoint metadata. It does not run rollout.
- Separate-async GSPO runs three steps with a policy synchronization interval
  of two, preserving the stale-policy snapshot and adapter-only weight-sync path.

These launchers accept an even `NUM_NPUS` of at least two. At eight NPUs,
GSPO and DPO use eight actor ranks; separate-async uses four trainer and four
rollout NPUs. Qwen3-Omni rollout TP stays at two because its tiny model has
four attention heads. Batch and dataset sizes scale with the actual actor
topology. Two-NPU separate-async uses one trainer and one TP=1 rollout rank.

The GSPO fixtures use attention head dimension 128 and matching MRoPE sections
for Ascend's inference attention kernel; the GPU default remains 32. These
math-only cases disable unused image/video/audio rollout profiling. Multimodal
offline DPO keeps its existing fixture and uses Transformers SDPA attention.

Run the new cases individually from the repository root:

```bash
NUM_NPUS=8 bash tests/special_e2e/run_flowgrpo_minimax_h3_npu.sh
NUM_NPUS=8 bash tests/special_e2e/run_qwen3_omni_npu.sh gspo
NUM_NPUS=8 bash tests/special_e2e/run_qwen3_omni_npu.sh offline-dpo
NUM_NPUS=8 bash tests/special_e2e/run_qwen3_omni_npu.sh separate-async
```

They also run through `run_npu_smoke_e2e.sh` and the combined runner. Use the
pinned dependencies and the model-specific extras required by the GPU cases,
including audio/video processing dependencies for multimodal DPO. The common
`tests/special_e2e/lib_npu_e2e.sh` sources CANN/ATB, validates visible NPUs,
and passes physical-device visibility to Ray workers. Model construction and
training run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Use `MINIMAX_H3_MODEL_PATH` and `MINIMAX_H3_DATA_DIR` for H3, or
`QWEN3_OMNI_MODEL_PATH`, `QWEN3_OMNI_MULTIMODAL_MODEL_PATH`, and
`QWEN3_OMNI_DATA_DIR` for Qwen3-Omni. These entrypoints use their own defaults
instead of inheriting a Qwen-Image `MODEL_PATH` or a different model's
`DATA_DIR`. Tiny builders reject unrelated existing output directories.

All four entries preserve nonzero exit codes, including native abort 134.
The compatibility exceptions for older tests are not applied to these new
workloads without matching runtime evidence. Record both completed training
steps and the final process exit code when validating on A2 or A3.

## Add a New NPU Smoke Test

### 1. Add the smallest useful entry point

Use a pytest file under tests/workers for rollout, worker, or engine coverage.
Use a shell script under tests/special_e2e for a trainer end-to-end path.

Keep tests small and reproducible:

- Use tiny-random checkpoints or weights cached on the CI runner.
- Use the smallest practical dataset, batch size, and training-step count.
- Do not require public model or dataset downloads at runtime.
- Preserve nonzero exit codes on failure.
- Do not reuse Ray processes left by another test.

### 2. Register the test in its group runner

Add core tests to run_npu_smoke_core.sh and trainer tests to
run_npu_smoke_e2e.sh. Assign the next unused numeric ID and call run_test:

    run_test 4 "my NPU smoke test"         env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"             NUM_NPUS="${NUM_NPUS}"         pytest -s tests/workers/test_my_npu_feature.py

Do not reuse an existing ID. Adding a test to an existing group needs no
workflow mapping change.

### 3. Configure devices

Group runners use eight NPUs by default and derive
ASCEND_RT_VISIBLE_DEVICES from --num-npus. Each command should receive both
ASCEND_RT_VISIBLE_DEVICES and NUM_NPUS. If a test requires a fixed number of
devices, validate it in the test entry point and fail with a clear message.

### 4. Update workflow groups and paths

The Resolve NPU CI groups step maps labels to groups:

    declare -A label_groups=(
      ["ready-for-ci"]="core e2e"
      ["ci-npu"]="core e2e"
      ["ci-npu-core"]="core"
      ["ci-npu-e2e"]="e2e"
    )

Multiple labels produce the union of their groups. The workflow invokes group
scripts in stable core, then e2e, order.

If the new test lives outside current path filters, add its path. Current
filters include verl_omni/**, tests/npu_smoke/**, tests/workers/**,
tests/special_e2e/**, pyproject.toml, .github/workflows/npu_smoke.yml,
and .github/vllm_omni_pin.txt.

### 5. Handle models and datasets

Prefer tiny checkpoints that can be generated and validated offline. Prepare
shared checkpoints in the workflow and model-specific checkpoints in their
test launch scripts. The Wan2.2 launcher prepares and validates its checkpoint
before training, so selecting only the core group does not prepare Wan2.2
weights.
Builders should reuse a valid cached checkpoint, rebuild missing or stale
components, and reload saved assets when file-presence checks are not
sufficient. Keep checkpoint paths overridable and do not overwrite shared
caches.

### 6. Validate locally

Run from the repository root in an environment with CANN, vLLM, vLLM-Ascend,
and vLLM-Omni installed.

Run all groups:

    bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 8

Run one group:

    bash tests/npu_smoke/run_npu_smoke_core.sh --num-npus 8
    bash tests/npu_smoke/run_npu_smoke_e2e.sh --num-npus 8

Select devices explicitly:

    ASCEND_RT_VISIBLE_DEVICES=0,2,4,6 NUM_NPUS=4       bash tests/npu_smoke/run_npu_smoke_core.sh

Before committing:

    bash -n tests/npu_smoke/*.sh
    pre-commit run --all-files

bash -n checks syntax only; it does not start Ray or allocate NPUs. Pre-commit
may reformat files, so review changes and rerun it until all hooks pass.

## CI Triggers

Pushes to main or v0.* run both groups. Pull requests targeting those branches
run only after a supported label is applied:

| Label | Requested groups |
|---|---|
| ready-for-ci | core e2e |
| ci-npu | core e2e |
| ci-npu-core | core |
| ci-npu-e2e | e2e |

Labels are not mutually exclusive. Applying both ci-npu-core and ci-npu-e2e
runs both groups.

## Runtime Environment

The current configuration uses eight NPUs on linux-aarch64-a2b4-8, a 120
minute timeout, 16 GiB shared memory, and this image:

    swr.cn-north-4.myhuaweicloud.com/mindspeed/pr-verl-omni-a2:v010-vllm030-omni030

The workflow prints CANN details, npu-smi output, relevant Ascend variables,
and installed vLLM component versions.

The FlowGRPO and DanceGRPO end-to-end launchers source CANN and ATB before
starting the workload:

```bash
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh
```

Set `ASCEND_HOME_PATH` to the toolkit directory for a custom installation.
If it is unset or empty, the launchers use the `latest` symlink. The selected
CANN version must remain compatible with the image's PyTorch and torch-npu.

The diagnostics and artifact-upload steps are defined directly in the workflow.
The NPU runner executes the job in a container, so repository-local composite
actions may otherwise be resolved against a different host workspace.

## Runtime Cleanup, Results, and Artifacts

Before each test, the shared library stops Ray, terminates stale
DiffusionWorker, VLLMWorker, and vLLMOmniHttpServer processes, waits five
seconds, force-kills remaining matches, and prints npu-smi information.

Per-test logs are stored under:

    logs/npu_smoke/<group>/<timestamp>/

Each test writes test_<id>.log and each group writes summary.log. The workflow
also writes group console logs under logs/npu_smoke/parallel/.

The final inline actions/upload-artifact step runs even after failure and
uploads logs/npu_smoke/** as the npu-smoke-logs-<job> artifact.

Some Ascend runtime versions can abort with exit code 134 during allocator
teardown after a successful workload. The runner accepts 134 only when the
test-specific positive completion markers are present. Earlier aborts, missing
markers, and other exit codes remain failures.

For Tests 0 and 3, the shared compatibility rule requires exactly one passed pytest case
with no failures, errors, or skips in the final summary, followed by the known
`corrupted size vs. prev_size` diagnostic and exit code 134. The runner emits a
warning and preserves the log. This handles the observed CI teardown failure;
it does not fix the underlying native memory issue.
