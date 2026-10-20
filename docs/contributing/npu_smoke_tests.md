# NPU Smoke Tests

Last updated: 10/20/2026.

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
| 4 | core | Diffusion agent loop, request and step execution | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_diffusion_single_turn_npu | 8 |
| 5 | core | Unseeded multi-worker rollout diversity | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_without_seed_npu | 8 |
| 6 | core | Seeded multi-worker rollout reproducibility | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_seed_reproducibility_npu | 8 |

ID 3 is reserved for the FSDP engine migration in PR #770, which is not part of
this branch. Do not reuse it. The GPU-to-NPU scope and remaining gaps are tracked
in [NPU smoke coverage](npu_smoke_coverage.md).

Tests 4-6 reuse the GPU tests' execution and assertions. Test 4 contains two
pytest cases (`request_execution` and `step_execution`); Tests 5 and 6 each
contain one. They require an even `NUM_NPUS` of at least 2, use rollout TP=2,
two agent-loop workers, 256x256 images, and a 10-step denoising schedule with
the same SDE window as the GPU seed test. Serial requests (`max_num_seqs=1`)
preserve the seed test's batch-shape constraint. Test 6 keeps the bit-equality
assertion; it does not substitute approximate equality on NPU.

Use the local checkpoint prepared by the workflow. The transformer head count
must be divisible by 2, and its `joint_attention_dim` must match the text
encoder's hidden size. The entry point reports missing hardware or assets as
failures. It does not rewrite the cached model or download weights at runtime.
The model can be prepared offline in a separate directory:

```bash
python3 tests/special_e2e/build_qwen_image_tiny_random.py \
    --output-dir "${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image"
```

After sourcing CANN and ATB, run just the new cases from the repository root:

```bash
export DEVICE_NAME=npu NUM_NPUS=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_NPU_SOCKET_PORT_RANGE=auto
python3 -m pytest -v -s tests/agent_loop/test_diffusion_agent_loop_npu.py
```

For a two-device local check, set both visibility variables to `0,1` and
`NUM_NPUS=2`. The existing core and combined runners also select Tests 4-6.
Each case owns a fresh local Ray cluster and restores the agent worker factory
afterward. Nonzero exits, including teardown aborts for these new IDs, remain
failures until their behavior is validated in the target environment.

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

    run_test 7 "my NPU smoke test"         env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"             NUM_NPUS="${NUM_NPUS}"         pytest -s tests/workers/test_my_npu_feature.py

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
filters include verl_omni/**, tests/npu_smoke/**, tests/agent_loop/**, tests/workers/**,
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
