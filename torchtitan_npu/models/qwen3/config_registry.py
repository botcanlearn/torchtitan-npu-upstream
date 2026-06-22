# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import ChatDataLoader
from torchtitan.models.qwen3.config_registry import qwen3_0_6b as _upstream_qwen3_0_6b
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

from torchtitan_npu.config.configs import CheckpointConfig, ParallelismConfig
from torchtitan_npu.converters import get_model_converter_config
from torchtitan_npu.models.qwen3 import model_registry


def sft_qwen3_30ba3b_math() -> Trainer.Config:
    def process_sample(sample):
        answer = sample["answer"]
        reasoning, final_answer = answer.rsplit("####", 1)
        return [
            {"role": "user", "content": sample["question"]},
            {
                "role": "assistant",
                "reasoning_content": reasoning.strip(),
                "content": final_answer.strip(),
            },
        ]

    model_spec = model_registry("30B-A3B")
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-30B-A3B",
        model_spec=model_spec,
        model_converters=ModelConvertersContainer.Config(
            converters=[
                get_model_converter_config("npu_rms_norm"),
                get_model_converter_config("npu_rope"),
                get_model_converter_config("npu_permute"),
                get_model_converter_config("npu_gmm"),
            ],
        ),
        optimizer=OptimizersContainer.Config(lr=1e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=4096,
            steps=100,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=4,
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="openai/gsm8k",
            load_dataset_kwargs={"name": "main", "split": "train"},
            sample_processor=process_sample,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
        ),
        checkpoint=CheckpointConfig(
            enable=True,
            folder="checkpoint",
            load_only=False,
            initial_load_in_hf=True,
            initial_load_path="./assets/hf/Qwen3-30B-A3B",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )


def qwen3_06b_test() -> Trainer.Config:
    config = _upstream_qwen3_0_6b()
    config.model_spec = model_registry("0.6B")
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            get_model_converter_config("npu_rms_norm"),
            get_model_converter_config("npu_rope"),
        ],
    )
    config.dataloader.dataset = "c4_test"
    config.lr_scheduler.warmup_steps = 20
    config.training.steps = 100
    config.checkpoint.initial_load_in_hf = True
    return config


def _qwen3_1_7b_converters() -> ModelConvertersContainer.Config:
    return ModelConvertersContainer.Config(
        converters=[],
    )


def _process_wordle_sample(sample: dict) -> list[dict]:
    """Convert a ``willcb/V3-wordle`` sample to Qwen3 chat messages.

    The dataset uses ``prompt`` (list of system + first-user messages) and
    ``completion`` (list of multi-turn assistant + user messages).
    We concatenate them into a single conversation and ensure ``content``
    is always a plain string.
    """
    if "messages" in sample:
        messages = sample["messages"]
    elif "prompt" in sample and "completion" in sample:
        messages = list(sample["prompt"]) + list(sample["completion"])
    else:
        raise KeyError(f"Wordle sample must have 'messages' or 'prompt'+'completion'. Got keys: {list(sample.keys())}")
    # Normalize: ensure content is always a string (Jinja2 chat templates
    # expect strings; some datasets store content as lists).
    for msg in messages:
        if isinstance(msg.get("content"), list):
            msg["content"] = "\n".join(str(c) for c in msg["content"])
    return messages


def sft_qwen3_1_7b_wordle() -> Trainer.Config:
    """SFT warmup for Qwen3-1.7B to play Wordle.

    Goal: 20 steps on ``willcb/V3-wordle`` to teach the model the
    environment's message format (multi-turn board-state → guess loop).

    Loads from HF base weights directly (no CPT prerequisite).
    Matches the prime-rl Wordle example SFT phase.

    Hardware: NGPU=1 (HF load safe), 64 GB NPU.
    """
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_registry("1.7B"),
        model_converters=_qwen3_1_7b_converters(),
        optimizer=OptimizersContainer.Config(lr=1e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            decay_ratio=1.0,
            decay_type="cosine",
            min_lr_factor=1.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            global_batch_size=64,
            seq_len=1024,
            max_norm=1.0,
            steps=20,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        dataloader=ChatDataLoader.Config(
            dataset_path="willcb/V3-wordle",
            load_dataset_kwargs={"split": "train"},
            sample_processor=_process_wordle_sample,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint_wordle_sft",
            load_only=False,
            initial_load_in_hf=True,
            initial_load_path="./assets/hf/Qwen3-1.7B",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
    )
