# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ascend port of the GPU composite agent-loop smoke (both reward-loop modes).

Reuses ``run_composite_single_turn`` from the GPU module so the AR+DiT batch
shapes, key contracts, reward score-split assertions and turn counts stay
identical; only the Ray cluster, device visibility, model path and attention
backends are adapted to the pinned Ascend stack.
"""

import gc
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def npu_composite_config(tmp_path_factory):
    """Create an isolated NPU Ray cluster and configure composite AR+DiT rollout."""
    import ray
    import torch
    import torch_npu  # noqa: F401
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager

    from tests.agent_loop.test_composite_agent_loop import _create_tp_compatible_model

    if not torch.npu.is_available():
        pytest.fail("These smoke tests require Ascend NPUs and torch-npu.")
    available = torch.npu.device_count()
    try:
        requested = int(os.getenv("NUM_NPUS", str(available)))
    except ValueError:
        pytest.fail("NUM_NPUS must be an integer.")
    if requested < 2 or requested > available or requested % 2:
        pytest.fail(f"TP=2 requires an even NUM_NPUS between 2 and {available}, got {requested}.")

    base_model_path = Path(
        os.getenv("MODEL_PATH", "~/.cache/modelscope/hub/models/tiny-random/Qwen-Image")
    ).expanduser()
    for component in (
        "model_index.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "vae/config.json",
        "scheduler/scheduler_config.json",
    ):
        if not (base_model_path / component).is_file():
            pytest.fail(f"Missing local Qwen-Image component: {base_model_path / component}")
    transformer = json.loads((base_model_path / "transformer/config.json").read_text())
    text_encoder = json.loads((base_model_path / "text_encoder/config.json").read_text())
    if transformer["num_attention_heads"] % 2:
        pytest.fail("The local tiny model needs a TP=2-compatible transformer; use build_qwen_image_tiny_random.py.")
    text_hidden_size = text_encoder.get("text_config", text_encoder)["hidden_size"]
    if transformer["joint_attention_dim"] != text_hidden_size:
        pytest.fail("Transformer joint_attention_dim must match the local text encoder hidden_size.")

    tmp_dir = str(tmp_path_factory.mktemp("qwen_image_model"))
    model_path = _create_tp_compatible_model(tmp_dir, str(base_model_path), num_attention_heads=2)

    config_dir = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="diffusion_trainer")
    config.trainer.device = "npu"
    config.trainer.n_gpus_per_node = requested
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.model.tokenizer_path = os.path.join(model_path, "tokenizer")
    config.actor_rollout_ref.model.algorithm = "dual_grpo"  # composite AR+DiT rollout pipeline
    config.actor_rollout_ref.model.attn_backend = "_native_npu"
    rollout = config.actor_rollout_ref.rollout
    rollout.name = "vllm_omni"
    rollout.mode = "async"
    rollout.enforce_eager = True
    rollout.step_execution = False
    rollout.nnodes = 1
    rollout.tensor_model_parallel_size = 2
    rollout.rollout_attn_backend = "TORCH_SDPA"
    rollout.load_format = "safetensors"
    rollout.n = 2  # dit samples per AR row
    rollout.m = 2  # ar part
    rollout.calculate_log_probs = True
    rollout.ar_calculate_log_probs = True  # ar part
    rollout.ar.response_length = 20  # ar part
    rollout.ar.temperature = 0.8  # ar part
    rollout.ar.top_k = 0  # ar part
    rollout.ar.top_p = 1.0  # ar part
    rollout.agent.num_workers = min(2, requested)
    rollout.agent.default_agent_loop = "composite_single_turn_agent"
    rollout.algo.noise_level = 1.0
    rollout.algo.sde_window_size = 2
    rollout.algo.sde_window_range = [0, 5]
    rollout.pipeline.height = 256
    rollout.pipeline.width = 256
    rollout.pipeline.num_inference_steps = 10
    rollout.pipeline.true_cfg_scale = 4.0
    rollout.pipeline.max_sequence_length = 1058
    rollout.max_model_len = 1058
    rollout.full_determinism = False

    config.reward.reward_manager.name = "MultiVisualRewardManager"
    # CompositeAgentLoopWorker requires AR weight 0 so MultiVisualRewardManager
    # does not double-count AR into the combined DiT score.
    OmegaConf.update(config, "reward.reward_functions.ar.weight", 0.0, force_add=True)
    config.data.max_prompt_length = 1058

    if ray.is_initialized():
        pytest.fail("Run these smoke cases in a process without an existing Ray connection.")
    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES") or ",".join(str(i) for i in range(requested))
    runtime_env = {
        "env_vars": {
            "DEVICE_NAME": "npu",
            "ASCEND_RT_VISIBLE_DEVICES": visible_devices,
            "ASCEND_VISIBLE_DEVICES": os.getenv("ASCEND_VISIBLE_DEVICES") or visible_devices,
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_NPU_SOCKET_PORT_RANGE": os.getenv("HCCL_NPU_SOCKET_PORT_RANGE") or "auto",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_LOGGING_LEVEL": "INFO",
        }
    }
    previous_device_name = os.environ.get("DEVICE_NAME")
    missing = object()
    previous_worker_class = getattr(AgentLoopManager, "agent_loop_workers_class", missing)
    os.environ["DEVICE_NAME"] = "npu"
    try:
        ray.init(address="local", runtime_env=runtime_env)
        if ray.cluster_resources().get("NPU", 0) < requested:
            pytest.fail(f"Ray must advertise at least {requested} NPU resources: {ray.cluster_resources()}")
        yield config
    finally:
        ray.shutdown()
        if previous_worker_class is missing:
            if hasattr(AgentLoopManager, "agent_loop_workers_class"):
                del AgentLoopManager.agent_loop_workers_class
        else:
            AgentLoopManager.agent_loop_workers_class = previous_worker_class
        if previous_device_name is None:
            os.environ.pop("DEVICE_NAME", None)
        else:
            os.environ["DEVICE_NAME"] = previous_device_name
        gc.collect()
        torch.npu.empty_cache()


@pytest.mark.parametrize("agent_reward_loop", [False, True])
def test_composite_single_turn_npu(npu_composite_config, agent_reward_loop: bool):
    """Preserve the GPU composite AR+DiT batch, key, score-split and turn assertions."""
    from tests.agent_loop.test_composite_agent_loop import run_composite_single_turn

    run_composite_single_turn(npu_composite_config, agent_reward_loop)
