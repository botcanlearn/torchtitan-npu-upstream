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
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.trainer import Trainer

from torchtitan_npu.config.configs import ParallelismConfig
from torchtitan_npu.converters.npu_registry import get_model_converter_config
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
        checkpoint=CheckpointManager.Config(
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
