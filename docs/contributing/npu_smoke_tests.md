# NPU Smoke Tests

Last updated: 10/27/2026.

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
| 4 | core | Diffusion agent loop, request and step execution | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_diffusion_single_turn_npu | 8 |
| 5 | core | Unseeded multi-worker rollout diversity | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_without_seed_npu | 8 |
| 6 | core | Seeded multi-worker rollout reproducibility | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_seed_reproducibility_npu | 8 |
| 7 | e2e | MiniMax-H3 T2VA FlowGRPO trainer | tests/special_e2e/run_flowgrpo_minimax_h3_npu.sh | 8 |
| 8 | e2e | Qwen3-Omni Thinker GSPO LoRA trainer | tests/special_e2e/run_qwen3_omni_npu.sh gspo | 8 |
| 9 | e2e | Qwen3-Omni multimodal offline DPO trainer | tests/special_e2e/run_qwen3_omni_npu.sh offline-dpo | 8 |
| 10 | e2e | Qwen3-Omni Thinker separate-async GSPO trainer | tests/special_e2e/run_qwen3_omni_npu.sh separate-async | 8 |
| 11 | core | Standalone vLLM reward engine and OCR scoring | tests/special_e2e/run_reward_engine_npu.sh | 8 |
| 12 | core | Rollout request batching and step-execution contracts | tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_contracts_npu.py | 8 |
| 13 | core | Diffusers FSDP/FSDP2 engine with sequence parallel 2 | tests/workers/test_diffusers_fsdp_engine_sp_npu.py | 8 |
| 14 | core | Diffusers Ulysses SP equivalence (torchrun) | tests/workers/test_diffusers_ulysses.py | 8 |
| 15 | core | Trainable FSDP2 snapshot restore (two-rank) | tests/workers/test_trainable_snapshot_gpu.py | 8 |
| 16 | core | Composite agent loop AR+DiT single turn | tests/agent_loop/test_composite_agent_loop_npu.py | 8 |
| 17 | core | OmniAgentLoopWorker image_gen tool agent loop | tests/agent_loop/test_omni_agent_loop_worker_npu.py | 8 |
| 18 | e2e | Qwen-Image FlowGRPO V1 sync trainer | tests/special_e2e/run_flowgrpo_qwen_image_v1_npu.sh | 8 |
| 19 | e2e | Qwen-Image-Edit FlowGRPO V1 sync trainer | tests/special_e2e/run_flowgrpo_qwen_image_edit_npu.sh | 8 |
| 20 | e2e | Qwen-Image online DPO V1 sync trainer | tests/special_e2e/run_online_dpo_qwen_image_npu.sh | 8 |
| 21 | e2e | DiffusionNFT V1 sync trainer | tests/special_e2e/run_diffusionnft_qwen_image_npu.sh | 8 |
| 22 | e2e | FlowGRPO V1 separate_async trainer | tests/special_e2e/run_flowgrpo_qwen_image_v1_separate_async_npu.sh | 8 |
| 23 | e2e | Diffusion OPD V1 sync colocated teachers | tests/special_e2e/run_diffusion_teacher_smoke_npu.sh (SMOKE=sync) | 8 |
| 24 | e2e | Diffusion OPD V1 separate_async standalone teachers | tests/special_e2e/run_diffusion_teacher_smoke_npu.sh (SMOKE=async) | 8 |
| 25 | e2e | Bagel PickScore LoRA FlowGRPO (v0 trainer) | tests/special_e2e/run_flowgrpo_bagel_pickscore_npu.sh | 8 |
| 26 | e2e | FlowGRPO synchronous separate trainer (v0) | tests/special_e2e/run_flowgrpo_qwen_image_separate_npu.sh | 8 |
| 27 | e2e | MiniCPM-o 4.5 Thinker GSPO LoRA (collocated sync, V1) | tests/special_e2e/run_gspo_minicpmo_thinker_lora_v1_npu.sh | 8 |
| 28 | core | Dual-adapter LoRA FSDP2 switch and EMA (FSDP1 deselected, deferred) | tests/workers/test_diffusers_fsdp_lora_adapter.py | 8 |
| 29 | e2e | Qwen3-TTS Talker full-parameter GRPO | tests/special_e2e/run_qwen3_tts_grpo_npu.sh | 8 |
| 30 | e2e | Named image previews and MiniMax-H3 offload placeholders | tests/trainer/diffusion/test_artifact_preview_dump_npu.py + tests/pipelines/test_minimax_h3_meta_output_npu.py | 8 |
| 31 | e2e | Boogu-Image DiffusionNFT trainer | tests/special_e2e/run_diffusionnft_boogu_image_npu.sh | 8 |
| 32 | core | Layered LoRA collection from nested FSDP units (Boogu) | tests/utils/test_fsdp_lora_collection_gpu.py | 8 |
| 33 | core | Diffusers VeOmni engine inference and training | tests/workers/test_diffusers_veomni_engine.py | 8 |

The combined runner has thirty-four entries: fifteen core (0, 3, 4, 5, 6,
11, 12, 13, 14, 15, 16, 17, 28, 32, 33) and nineteen e2e (1, 2, 7, 8, 9,
10, 18-27, 29, 30, 31).
Entry 4 is parametrized with two pytest cases (`request_execution` and
`step_execution`), entry 11 runs four pytest cases, entry 12 runs five,
entry 13 runs two, entry 14 runs ten on eight devices, entry 16 runs two
(with and without the agent reward loop), entry 28 runs its two fsdp2
params with the two fsdp params explicitly deselected (see the deferred
FSDP1 note in the validation status section), and entry 32 runs two, so
the core group runs thirty-six pytest cases in total.

Test 3 ports the `fsdp` case of `tests/workers/test_diffusers_fsdp_engine.py`
(GPU core ID 2). It reuses the GPU test's synthetic trajectories and engine
inference/training exercise, with `_native_npu` attention and sequence parallel
size 1. It requires at least two visible NPUs and a local tiny Qwen-Image model.
FSDP2 and sequence-parallel execution are covered by Test 13 below, not by this
case. The test checks finite inference log probabilities, training loss, and
gradient norm; it does not establish GPU/NPU numerical parity or model
convergence.

Run only the migrated case from the repository root after sourcing CANN/ATB:

```bash
DEVICE_NAME=npu NUM_NPUS=2 \
MODEL_PATH="${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image" \
python3 -m pytest -v -s tests/workers/test_diffusers_fsdp_engine_npu.py
```

The same case runs through `run_npu_smoke_core.sh` and the combined
`run_npu_smoke_tests.sh` entry point. Missing hardware or model files fail
explicitly rather than producing an all-skipped smoke success.

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
afterward. Nonzero exits remain failures except for the narrowly classified
post-pytest teardown abort described below.

### Standalone reward engine

Test 11 ports the whole GPU file `tests/reward_loop/test_visual_reward_manager.py`
(GPU core ID 5) as four pytest cases at file granularity. The GenRM case starts
reward-model servers through `RewardLoopManager` with a dedicated resource pool
and `rollout.name=vllm`; no actor rollout server is created.
A local random Qwen2.5-VL checkpoint exercises the architecture involved in
issue #791 without downloading the full reward model. Each launch uses a new,
test-owned `VLLM_CACHE_ROOT`, so existing architecture-inspection caches cannot
hide cold-start failures.

At eight NPUs the GenRM case starts four TP=2 reward replicas, checks each
`/v1/models` endpoint, and scores three synthetic images through the reward
router and the existing OCR scorer. It requires finite `(3, 1)` scores in
`[0, 1]` and one model response per image. Random weights do not test OCR
quality or convergence. The dedicated servers stay resident
(`free_cache_engine=False`); reward-model offload and sleep/wake behavior are
outside this case.

The two determinism cases port the GPU bitwise-reproducibility contract:
`full_determinism=True` with a fixed seed, two independent server runs per case,
and `torch.equal` on `rm_scores` plus exact string equality on
`genrm_response`. The single-replica case uses one TP=2 replica and the
multi-replica case uses two TP=1 replicas, matching the GPU smoke topology.
Sample images are sized for the tiny Ascend fixture (56x56 with the standalone
case's `mm_processor_kwargs`); the bit-equality assertions are unchanged.

The rule-reward case keeps `reward_model.enable=False` and scores
`jpeg_compressibility` samples through the CPU-side scorer with the tiny
Qwen-Image tokenizer; it adds finite-score assertions to the GPU case's
print-only contract. All nonzero exits, including native abort 134, remain
failures for this entry.

```bash
NUM_NPUS=8 bash tests/special_e2e/run_reward_engine_npu.sh
```

The launcher builds its own checkpoint under
`~/.cache/verl-omni-npu-smoke/Qwen2.5-VL-Reward-NPU`. Set
`NPU_REWARD_MODEL_PATH` to select another empty output directory owned by this
test. Existing unrelated or incomplete directories are rejected rather than
overwritten. Keep the final pytest result and process exit code when reporting
whether the cold reward-server path succeeds on a given environment.

On an eight-device Ascend 910B2 server, this case passed in 178.83 seconds with
exit 0 using the image's jemalloc preload from `/root/.bashrc`. Noninteractive
shells do not load that file automatically. In fresh-cache `ModelConfig` probes,
removing `LD_PRELOAD` reproduced the #791 architecture-inspection failure in
three of three runs; explicitly preloading jemalloc passed three of three.
These results identify an environment-sensitive failure mode, not a repair of
the underlying native heap corruption. The launcher does not select an allocator
or suppress failures; retain the allocator setup when recording run conditions.

### Rollout contracts and FSDP2/SP engine cases

Test 12 ports the remaining cases of
`tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py` (GPU core
ID 1) that Test 0 does not cover: request-level batching with named media
artifacts (`output_type` image and latent, four concurrent prompts under
`max_num_seqs>1` plus `request_batch_max_wait_ms`), and the FlowGRPO,
DiffusionNFT, and online-DPO `step_execution=True` trajectory contracts. The
contract assertions are imported verbatim from the GPU module, so both devices
enforce identical output contracts (artifact specs and dtypes, CPU-resident
artifacts, `visual_reward_frames` projection, trajectory shape relations,
fp32 `latents_clean`, and preserved `None` negative-embed keys). Each server
runs TP=1 on a single NPU, mirroring the GPU topology; `ray.get` timeouts are
relaxed from 600s to 900s for the slower device. The upstream-skipped GPU
`test_generate` (mixed-logprobs batching) stays out of scope on both sides.

Test 13 closes the two dimensions of GPU core ID 2 that Test 3 pins away:
`strategy` in {`fsdp`, `fsdp2`} crossed with `ulysses_sequence_parallel_size=2`
at `fsdp_size=NUM_NPUS`. It reuses the GPU test's SP-compatible heads=2 model
fixture, config builder, and engine batches, and keeps Test 3's contract
assertions (finite inference log probabilities, training loss, and gradient
norm). The actor uses Diffusers `native` SDPA because the NPU-specific
context-parallel attention backend does not implement backward in the pinned
stack. Numerical parity with the GPU run is not established.

Test 14 runs the shared GPU file `tests/workers/test_diffusers_ulysses.py`
(GPU core ID 8a) unmodified in body: device detection and the distributed
backend now come from `verl.utils.device` (`get_device_name` /
`get_torch_device` / `get_nccl_backend`), which keeps GPU behavior identical
and selects `hccl` plus `torch.npu` on Ascend. Under `torchrun` with eight
NPUs all ten parametrized cases execute (SP=2 and SP=4 forward, forward +
backward, MiniMax-H3 aligned/unaligned all-task, and FSDP-wrapped SP
equivalence); the GPU two-device runner self-skips the five SP=4 cases.
Equivalence tolerances are the GPU originals — this is same-device SP-vs-
non-SP equivalence, not cross-device numerical parity.

Test 15 runs the shared GPU file `tests/workers/test_trainable_snapshot_gpu.py`
(GPU core ID 8b) unmodified in body: seeding, synchronization, device selection
and the distributed backend now come from `verl.utils.device`, which keeps GPU
behavior identical and selects `hccl` plus `torch.npu` on Ascend. The pytest
entry spawns its own two-rank `torch.distributed.run` subprocess over the first
two visible devices and checks per-rank JSON verdicts: trainable-only vs full
snapshot bit-equality across save/restore/train cycles, CPU-offload policy
handling, and rejection of injected rank-local save/restore/layout/clone
failures on every rank. `REQUIRE_TRAINABLE_SNAPSHOT_GPU=1` turns a missing
two-device environment into a failure instead of a skip. All nonzero exits,
including native abort 134, remain failures for entries 12-15.

```bash
DEVICE_NAME=npu NUM_NPUS=8 pytest -s tests/workers/test_diffusers_fsdp_engine_sp_npu.py
DEVICE_NAME=npu NUM_NPUS=8 python3 -m pytest -s \
    tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_contracts_npu.py
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --local-ranks-filter=0 tests/workers/test_diffusers_ulysses.py
REQUIRE_TRAINABLE_SNAPSHOT_GPU=1 python3 -m pytest -s \
    tests/workers/test_trainable_snapshot_gpu.py
```

### Composite and tool agent loops

Test 16 ports `tests/agent_loop/test_composite_agent_loop.py` (GPU core ID 6,
both `agent_reward_loop` modes) and Test 17 ports
`tests/agent_loop/test_omni_agent_loop_worker_gpu.py` (GPU core ID 7). Both
GPU modules expose their assertion bodies as `run_*` functions that the NPU
tests call directly, so the AR+DiT batch shapes, trajectory and media-artifact
keys, fake-reward score-split contract (`reward/ar` 1.0 vs combined 1.5), turn
counts, tool-loop rollout stamps, agentic metrics and trajectory dumps are
asserted identically. The NPU fixtures own the Ray cluster, device visibility
and attention backends (`_native_npu` actor + `TORCH_SDPA` rollout for the
Qwen-Image composite; thinker-only Qwen3-Omni with head dim 128, eager mode
and `weight_nz_mode=0` for the tool loop). Test 17 reuses the GPU fake
diffusion HTTP tool server and the `vllm_omni` rollout registration hook
unchanged.

### MiniMax-H3 and Qwen3-Omni model smoke

Test 7 reuses GPU diffusion E2E ID 7: tiny MiniMax-H3 text-to-video-and-audio
FlowGRPO, one V1 sync training step, FSDP2 with actor sequence parallel size 2
and rollout tensor parallel size 2. Its offline builder generates deterministic
random weights with the pinned GPU fixture's geometry and reuses the existing
tiny VAE stubs and checkpoint conversion. The generated weights differ from
the downloaded GPU fixture; this tests model execution, not output quality or
GPU/NPU numerical parity. FL2VA, Ref2VA, and standalone Ulysses equivalence
checks are outside this case.
The actor uses Diffusers `native` SDPA because the NPU-specific context-parallel
attention backend does not implement backward in the pinned stack.

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
The online NPU actor keeps entropy eager: dynamic-shape compiled entropy
triggered a vector-core failure when token counts changed in the tested stack.
BF16 actor computation and the original training and synchronization steps remain.

Training workers normalize nested multi-axis position IDs before minibatch
selection. This preserves token lengths when TransferQueue infers the wrong
jagged axis for equal-shaped samples; changing the axis flag alone is insufficient.

A weights-only wake keeps rollout admission paused until all sleeping engine
tags are restored, preventing request failures during asynchronous weight sync.

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

### Diffusion E2E trainer matrix and LoRA adapter cases

Tests 18-26 port the GPU diffusion E2E group (IDs 0-6, 8-10) as thin NPU
wrappers around the unmodified GPU launchers: `lib_npu_e2e.sh` supplies
CANN/ATB sourcing, visible-device validation and the `trainer.device=npu`
plus Ray-visibility runtime args, and each wrapper pins model-specific paths.
The GPU launchers select native/TORCH_SDPA attention automatically because
`fa_available()` is false on the pinned Ascend stack. Notes per entry:

- 18/20 (FlowGRPO V1 sync, online DPO): the OCR GenRM reward model defaults
  to the tiny random Qwen2.5-VL checkpoint from Test 11 (the architecture
  validated on Ascend in #791 probes) instead of the GPU smoke's tiny
  qwen3-vl; the weighted multi-reward and pairing contracts are unchanged.
  Test 18 is the V1 counterpart of Test 1 — Test 1 runs the deprecated v0
  `main_diffusion` trainer and does not cover the V1 sync path.
- 19 (Qwen-Image-Edit): the tiny edit checkpoint is built offline by the
  wrapper before delegating to the GPU launcher.
- 22/26 (separate_async, v0 separate): keep the launcher default
  `CKPT_BACKEND=nccl`, which verl resolves to its `HCCLCheckpointEngine`
  on Ascend (the CUDA/NPU checkpoint engines register under the same
  name and are mutually exclusive by construction; the cupy-based NCCL
  module simply fails to import there). At eight NPUs the default split
  is four actor ranks plus four standalone TP=1 rollout replicas.
- 23/24 (OPD teachers): reuse the GPU launcher's offline tiny SD3 builds
  (student + two teachers, three seeds) and its SMOKE=sync/async modes.
- 25 (Bagel PickScore): deprecated v0 trainer kept covered until the Bagel
  v1 recipe lands upstream; BAGEL-MoT and PickScore tiny checkpoints are
  built offline by the GPU launcher.

Test 27 ports GPU omni E2E ID 4 (MiniCPM-o 4.5 thinker GSPO LoRA, collocated
sync V1, frozen vpm/apm towers via exclude_modules + FSDP2 ignored params).
Rollout TP stays at 2 on eight NPUs while the actor uses all visible ranks.
The builder downloads the pinned MiniCPM-o remote-code source checkpoint on
first use; pre-cache it once with HF online because test runs execute with
`HF_HUB_OFFLINE=1`.

Test 30 ports GPU diffusion E2E ID 11 (the device-residency regressions
`test_artifact_preview_dump.py` and `test_minimax_h3_meta_output.py`): NPU
copies assert the same PNG dump sizes/pixels, latent immutability, H3 meta
placeholder pass-through and exact prompt-embed/latent values with `npu`
device placement, reusing the CPU-module helpers.

Test 29 ports GPU omni E2E ID 3 (Qwen3-TTS Talker full-parameter GRPO, two
updates, in-process CPU reward, exact global_step completion check). It
requires the GPU smoke prepare parity installs in the environment — the
pinned `qwen-tts` archive with `--no-deps`, `sox` and the audio extras — and
the pinned tiny-random-qwen3-tts source revision pre-cached in the HF cache,
because runs execute with `HF_HUB_OFFLINE=1`.

Test 28 runs the shared GPU file
`tests/workers/test_diffusers_fsdp_lora_adapter.py` (not wired into any GPU
runner) after device-API-ization: accelerator detection and device counts
come from `verl.utils.device`, so the dual-adapter cases execute on Ascend
with `DEVICE_NAME=npu`. The two fsdp2 params pass; the two fsdp (FSDP1)
params are explicitly deselected in the runner because on NPU the
first-collected adapter loses its nested `transformer_blocks` LoRA keys
under FSDP1 while the identical fsdp2 assertions pass — recorded as a
deferred production-side investigation with `-vv` key-diff evidence, not
skipped inside the test and not relaxed.
`tests/utils/gpu_test_topology.py` was device-API-ized the same way; GPU
behavior is unchanged.

Test 31 ports GPU diffusion E2E ID 10 (Boogu-Image DiffusionNFT, v0 trainer,
MODE=t2i). The GPU launcher's CUDA-only preflight now accepts torch_npu when
the CUDA build is absent; GPU behavior is unchanged. The environment needs
the `boogu-image` package and OpenCV, and the builder's processor/scheduler
source assets pre-cached from `Boogu/Boogu-Image-0.1-Base` because runs
execute with `HF_HUB_OFFLINE=1`.

Test 32 runs the Boogu-based layered LoRA collection regression
(`tests/utils/test_fsdp_lora_collection_gpu.py`, single-rank, not wired into
any GPU runner) after the same device-API-ization; it needs `boogu-image`.

Test 33 runs the shared VeOmni engine test (`test_diffusers_veomni_engine.py`,
GPU core ID 3) after device-API-ization; it needs `veomni==0.1.12` installed
`--no-deps` exactly as the GPU smoke prepare action does. VeOmni's NPU actor
support is still being established upstream (#812); a failure here is
recorded with its stack rather than skipped.

Entry 14 initially failed all six Qwen-Image cases on Ascend with
`aclnnRmsNorm` shape errors: the shared GPU test derived the text-branch
width from `cross_attention_dim`/`encoder_hidden_size` with a hardcoded 32
fallback, while the tiny Ascend checkpoint carries only
`joint_attention_dim=16`; the fallback chain now ends at `joint_attention_dim`
(GPU fixtures keep their existing keys, so GPU behavior is unchanged). The
same change makes the file's `__main__` propagate the pytest exit code —
previously `pytest.main(...)` returned without `sys.exit`, so a torchrun
launch exited 0 even when cases failed.

### Validation status of the combined suite

Entries 0-6 and 11 passed together as an eight-entry suite on an 8x Ascend
910B2 server at PR #789 head `2c2f23f` (core, e2e, and combined raw exit
codes 0; no teardown-abort classification was used); entry 11 covered only
the GenRM case there, and its three additional reward cases are new on this
branch. Entries 7-10 passed together with entries 0-3 as a separate
eight-entry suite on the same class of server from the model-smoke branch.
Entries 12-28 are new on this branch. Entry 12 passed standalone on an 8x
Ascend 910B2 server at `97ed9b2` (5 passed, raw exit 0) and entry 13 at
`f942088` (2 passed, raw exit 0). Entry 14's first run exposed the
fixture-width and exit-code defects described above; it is re-validated
after the fix. Entries 15 and 17 passed standalone on the 8x910B2 server at
`c8a1b71` (raw exit 0 each); entry 16's first run exposed the CUDA-only
`fork_rng` production bug fixed in the dual-GRPO adapter and is re-validated
with that fix. Entry 28 passes on fsdp2; its two fsdp (FSDP1) cases hit a
`collect_lora_params` old-vs-default key mismatch on NPU that is recorded for
follow-up instead of being skipped or relaxed. Entries 18-27 and 29-33 are
pending hardware validation, and the full thirty-four-entry combined suite
has not yet been validated in a single run.

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
run_npu_smoke_e2e.sh. Assign an unused numeric ID and call run_test:

    run_test 14 "my NPU smoke test" \
        env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
            NUM_NPUS="${NUM_NPUS}" \
        pytest -s tests/workers/test_my_npu_feature.py

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

For Tests 0, 3, 5, and 6, the shared compatibility rule requires exactly one
passed pytest case; Test 4 requires exactly two. It also requires no failures,
errors, or skips in the final summary, only the known
`corrupted size vs. prev_size` diagnostic after that summary, and exit code
134. The runner emits a warning and preserves the log. This handles the
observed CI teardown failure; it does not fix the underlying native memory
issue.
