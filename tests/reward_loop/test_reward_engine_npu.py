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
"""Real standalone vLLM reward-server startup and OCR scoring on Ascend."""

import os
from pathlib import Path

import pytest


def test_standalone_reward_engine_npu():
    """Port the GPU GenRM path without starting an actor rollout server."""
    import ray
    import requests
    import torch
    import torch_npu  # noqa: F401
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from verl.experimental.reward_loop import RewardLoopManager
    from verl.protocol import DataProto
    from verl.utils import hf_tokenizer
    from verl.workers.rollout.replica import RolloutMode

    if not torch.npu.is_available():
        pytest.fail("The standalone reward smoke requires Ascend NPUs.")
    requested = int(os.environ.get("NUM_NPUS", "8"))
    if requested < 2 or requested % 2 or requested > torch.npu.device_count():
        pytest.fail("NUM_NPUS must be even, at least two, and no greater than the visible NPU count.")
    if ray.is_initialized():
        pytest.fail("Run this test without an existing Ray connection.")
    model_path = Path(os.environ["NPU_REWARD_MODEL_PATH"]).expanduser().resolve()
    assert (model_path / "tiny_reward_checkpoint_metadata.json").is_file()
    assert os.environ.get("VLLM_CACHE_ROOT"), "Use the launcher to isolate model-inspection caches."

    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "verl_omni/trainer/config"), version_base="1.3"):
        config = compose(config_name="diffusion_trainer")
    config.trainer.device = "npu"
    config.trainer.n_gpus_per_node = requested
    config.actor_rollout_ref.model.path = str(model_path)
    config.actor_rollout_ref.model.tokenizer_path = str(model_path)
    config.reward.num_workers = 1
    config.reward.custom_reward_function.path = str(repo_root / "verl_omni/utils/reward_score/genrm_ocr.py")
    config.reward.custom_reward_function.name = "compute_score_ocr"
    reward = config.reward.reward_model
    reward.enable = True
    reward.enable_resource_pool = True
    reward.n_gpus_per_node = requested
    reward.nnodes = 1
    reward.model_path = str(model_path)
    rollout = reward.rollout
    rollout.name = "vllm"
    rollout.tensor_model_parallel_size = 2
    rollout.dtype = "bfloat16"
    rollout.load_format = "safetensors"
    rollout.gpu_memory_utilization = 0.25
    rollout.enforce_eager = True
    rollout.free_cache_engine = False
    rollout.skip_tokenizer_init = False
    rollout.max_model_len = 512
    rollout.max_num_batched_tokens = 512
    rollout.max_num_seqs = 1
    rollout.prompt_length = 256
    rollout.response_length = 8
    rollout.enable_prefix_caching = False
    with open_dict(rollout.engine_kwargs):
        rollout.engine_kwargs.vllm = {
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "mm_processor_kwargs": {"min_pixels": 56 * 56, "max_pixels": 112 * 112},
        }

    env_vars = {
        key: os.environ[key]
        for key in (
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "VLLM_CACHE_ROOT",
            "PYTHONPATH",
        )
        if key in os.environ
    }
    env_vars.update(
        {
            "DEVICE_NAME": "npu",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_ASCEND_ENABLE_NZ": "0",
        }
    )
    ray.init(
        address="local",
        num_cpus=min(os.cpu_count() or 16, 32),
        include_dashboard=False,
        runtime_env={"env_vars": env_vars},
    )
    try:
        assert ray.cluster_resources().get("NPU", 0) >= requested
        manager = RewardLoopManager(config)
        engine = manager.reward_model_manager
        assert engine is not None and engine.resource_pool is None
        assert len(engine.rollout_replicas) == requested // 2
        for replica in engine.rollout_replicas:
            assert replica.is_reward_model and replica.rollout_mode == RolloutMode.STANDALONE
        with requests.Session() as session:
            session.trust_env = False
            for address in engine.server_addresses:
                response = session.get(f"http://{address}/v1/models", timeout=30)
                response.raise_for_status()
                assert str(model_path) in {entry["id"] for entry in response.json()["data"]}

        tokenizer = hf_tokenizer(str(model_path))
        prompt = torch.tensor(tokenizer.encode("Read the text in this image."), dtype=torch.long)
        batch_size = 3
        data = DataProto.from_dict(
            tensors={
                "input_ids": prompt.unsqueeze(0).repeat(batch_size, 1),
                "responses": (torch.arange(batch_size * 3 * 56 * 56) % 256)
                .to(torch.uint8)
                .reshape(batch_size, 3, 56, 56),
            },
            non_tensors={
                "data_source": ["ocr"] * batch_size,
                "reward_model": [{"ground_truth": "OCR"}] * batch_size,
                "media_kind": ["image"] * batch_size,
                "extra_info": [{} for _ in range(batch_size)],
            },
        )
        output = manager.compute_rm_score(data)
        scores = output.batch["rm_scores"]
        assert scores.shape == (batch_size, 1)
        assert torch.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        responses = output.non_tensor_batch["genrm_response"]
        assert len(responses) == batch_size and all(isinstance(text, str) for text in responses)
        print(f"Standalone reward replicas: {len(engine.rollout_replicas)}; scored images: {batch_size}")
        print(f"OCR scores: {scores.flatten().tolist()}; model responses: {responses.tolist()}")
    finally:
        ray.shutdown()


_NPU_REWARD_ENV_PASSTHROUGH = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "VLLM_CACHE_ROOT",
    "PYTHONPATH",
)


def _npu_reward_env_vars() -> dict:
    """Ray runtime env vars for reward-only runs (mirrors the standalone test)."""
    env_vars = {key: os.environ[key] for key in _NPU_REWARD_ENV_PASSTHROUGH if key in os.environ}
    env_vars.update(
        {
            "DEVICE_NAME": "npu",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_ASCEND_ENABLE_NZ": "0",
        }
    )
    return env_vars


def _require_reward_npus(min_npus: int) -> int:
    """Validate the NPU environment and return the requested device count."""
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        pytest.fail("The reward smoke requires Ascend NPUs.")
    try:
        requested = int(os.environ.get("NUM_NPUS", "8"))
    except ValueError:
        pytest.fail("NUM_NPUS must be an integer.")
    if requested < min_npus or requested > torch.npu.device_count():
        pytest.fail(f"Need {min_npus} <= NUM_NPUS <= visible NPU count, got {requested}.")
    assert os.environ.get("VLLM_CACHE_ROOT"), "Use the launcher to isolate model-inspection caches."
    return requested


def _resolve_reward_npu_topology(num_replicas: int) -> tuple[int, int]:
    """Return (n_gpus_per_node, tp_size) preserving the GPU smoke semantics.

    A single replica spans TP=2; multiple replicas use TP=1 with one device
    each, matching ``_build_deterministic_reward_config`` on the GPU side.
    """
    if num_replicas == 1:
        return 2, 2
    return num_replicas, 1


def _build_deterministic_reward_config_npu(*, seed: int, n_gpus_per_node: int, tp_size: int):
    """Compose the standalone reward config with vLLM full determinism enabled."""
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    model_path = Path(os.environ["NPU_REWARD_MODEL_PATH"]).expanduser().resolve()
    assert (model_path / "tiny_reward_checkpoint_metadata.json").is_file()
    repo_root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo_root / "verl_omni/trainer/config"), version_base="1.3"):
        config = compose(config_name="diffusion_trainer")
    config.trainer.device = "npu"
    config.trainer.n_gpus_per_node = n_gpus_per_node
    config.actor_rollout_ref.model.path = str(model_path)
    config.actor_rollout_ref.model.tokenizer_path = str(model_path)
    config.reward.custom_reward_function.path = str(repo_root / "verl_omni/utils/reward_score/genrm_ocr.py")
    config.reward.custom_reward_function.name = "compute_score_ocr"
    reward = config.reward.reward_model
    reward.enable = True
    reward.enable_resource_pool = True
    reward.n_gpus_per_node = n_gpus_per_node
    reward.nnodes = 1
    reward.model_path = str(model_path)
    rollout = reward.rollout
    rollout.name = "vllm"
    rollout.tensor_model_parallel_size = tp_size
    rollout.dtype = "bfloat16"
    rollout.load_format = "safetensors"
    rollout.gpu_memory_utilization = 0.25
    rollout.enforce_eager = True
    rollout.free_cache_engine = False
    rollout.skip_tokenizer_init = False
    rollout.max_model_len = 512
    rollout.max_num_batched_tokens = 512
    rollout.max_num_seqs = 1
    rollout.prompt_length = 256
    rollout.response_length = 8
    rollout.enable_prefix_caching = False
    # The determinism contract mirrors the GPU case; sizes are adapted to the
    # tiny Ascend fixture (cf. the standalone test's engine kwargs).
    rollout.full_determinism = True
    rollout.seed = seed
    with open_dict(rollout.engine_kwargs):
        rollout.engine_kwargs.vllm = {
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "mm_processor_kwargs": {"min_pixels": 56 * 56, "max_pixels": 112 * 112},
        }
    return config


def _create_npu_ocr_data_samples(model_path: Path, *, batch_size: int = 3, seed: int = 42):
    """Seeded OCR samples mirroring the GPU determinism fixture at NPU size."""
    import torch
    from verl.protocol import DataProto
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(str(model_path))
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.tensor(tokenizer.encode("Read the text in this image."), dtype=torch.long)
    responses = torch.randint(0, 256, (batch_size, 3, 56, 56), dtype=torch.uint8, generator=generator)
    return DataProto.from_dict(
        tensors={
            "input_ids": prompt.unsqueeze(0).repeat(batch_size, 1),
            "responses": responses,
        },
        non_tensors={
            "data_source": ["ocr"] * batch_size,
            "reward_model": [{"ground_truth": "OCR"}] * batch_size,
            "media_kind": ["image"] * batch_size,
            "extra_info": [{} for _ in range(batch_size)],
        },
    )


def _run_rm_with_seed_npu(seed: int, data, *, num_replicas: int):
    """Score ``data`` with a fresh NPU reward server so runs are independent."""
    import ray
    from verl.experimental.reward_loop import RewardLoopManager
    from verl.workers.rollout.replica import RolloutMode

    n_gpus_per_node, tp_size = _resolve_reward_npu_topology(num_replicas)
    ray.init(
        address="local",
        num_cpus=min(os.cpu_count() or 16, 32),
        include_dashboard=False,
        runtime_env={"env_vars": _npu_reward_env_vars()},
    )
    try:
        assert ray.cluster_resources().get("NPU", 0) >= n_gpus_per_node
        config = _build_deterministic_reward_config_npu(seed=seed, n_gpus_per_node=n_gpus_per_node, tp_size=tp_size)
        manager = RewardLoopManager(config)
        engine = manager.reward_model_manager
        assert engine is not None and engine.resource_pool is None
        assert len(engine.rollout_replicas) == num_replicas
        for replica in engine.rollout_replicas:
            assert replica.is_reward_model and replica.rollout_mode == RolloutMode.STANDALONE
        return manager.compute_rm_score(data)
    finally:
        ray.shutdown()


def _assert_determinism_npu(num_replicas: int):
    """Same seed across two independent NPU reward runs -> bitwise-aligned outputs."""
    import torch

    n_gpus_per_node, _ = _resolve_reward_npu_topology(num_replicas)
    _require_reward_npus(n_gpus_per_node)
    model_path = Path(os.environ["NPU_REWARD_MODEL_PATH"]).expanduser().resolve()
    data = _create_npu_ocr_data_samples(model_path)

    run1 = _run_rm_with_seed_npu(42, data, num_replicas=num_replicas)
    run2 = _run_rm_with_seed_npu(42, data, num_replicas=num_replicas)
    for idx, (o1, o2) in enumerate(zip(run1, run2, strict=False)):
        assert torch.equal(o1.batch["rm_scores"], o2.batch["rm_scores"]), (
            f"[replicas={num_replicas}] same-seed rm_scores[{idx}] differ: "
            f"{o1.batch['rm_scores']} vs {o2.batch['rm_scores']}"
        )
        assert o1.non_tensor_batch["genrm_response"] == o2.non_tensor_batch["genrm_response"], (
            f"[replicas={num_replicas}] same-seed genrm_response[{idx}] differs: "
            f"{o1.non_tensor_batch['genrm_response']} vs {o2.non_tensor_batch['genrm_response']}"
        )
    print(f"NPU reward determinism verified across two runs with {num_replicas} replica(s).")


def test_deterministic_reward_reproducibility_single_replica_npu():
    """Port of the GPU single-replica bitwise determinism case (1 replica, TP=2)."""
    _assert_determinism_npu(num_replicas=1)


def test_deterministic_reward_reproducibility_multi_replica_npu():
    """Port of the GPU multi-replica bitwise determinism case (2 replicas, TP=1)."""
    _assert_determinism_npu(num_replicas=2)


def test_rule_reward_npu():
    """Port of the GPU rule-reward case: CPU-side scorer, no reward-model server."""
    import ray
    import torch
    import torch_npu  # noqa: F401
    from hydra import compose, initialize_config_dir
    from verl.experimental.reward_loop import RewardLoopManager
    from verl.utils import hf_tokenizer

    from tests.reward_loop.test_visual_reward_manager import create_data_samples

    _require_reward_npus(2)
    rollout_model = Path(os.getenv("MODEL_PATH", "~/.cache/modelscope/hub/models/tiny-random/Qwen-Image")).expanduser()
    for component in ("model_index.json", "tokenizer/tokenizer_config.json"):
        if not (rollout_model / component).is_file():
            pytest.fail(f"Missing local Qwen-Image component: {rollout_model / component}")
    if ray.is_initialized():
        pytest.fail("Run this test without an existing Ray connection.")

    repo_root = Path(__file__).resolve().parents[2]
    ray.init(
        address="local",
        num_cpus=min(os.cpu_count() or 16, 32),
        include_dashboard=False,
        runtime_env={"env_vars": _npu_reward_env_vars()},
    )
    try:
        with initialize_config_dir(config_dir=str(repo_root / "verl_omni/trainer/config"), version_base="1.3"):
            config = compose(config_name="diffusion_trainer")
        config.trainer.device = "npu"
        config.trainer.n_gpus_per_node = int(os.environ.get("NUM_NPUS", "8"))
        config.actor_rollout_ref.model.path = str(rollout_model)
        config.actor_rollout_ref.model.tokenizer_path = str(rollout_model / "tokenizer")
        config.reward.num_workers = 1
        config.reward.reward_model.enable = False

        reward_loop_manager = RewardLoopManager(config)
        rollout_tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path)
        data = create_data_samples(rollout_tokenizer, data_source="jpeg_compressibility")
        outputs = reward_loop_manager.compute_rm_score(data)
        for output in outputs:
            scores = output.batch["rm_scores"]
            assert scores.numel() > 0, "rule reward must produce a score per sample"
            assert torch.isfinite(scores).all(), f"nonfinite rule reward scores: {scores}"
            print(f"Rule-based Reward Score:\n{scores}\n")
    finally:
        ray.shutdown()
