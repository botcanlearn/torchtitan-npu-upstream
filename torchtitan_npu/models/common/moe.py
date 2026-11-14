# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared configuration checks for NPU MoE dispatchers."""

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.base.model import BaseModel
from torchtitan.models.common.moe.token_dispatcher import (
    BaseEPTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)


def dispatcher_capacity(*, num_tokens: int, parallelism: ParallelismConfig) -> int:
    """Return the per-rank capacity required by an EP dispatcher.

    CP and TP/SP shard the token dimension before dispatch.  DeepEP and
    HybridEP therefore need a capacity that covers the resulting local token
    count, and the token budget must divide evenly across those axes.
    """
    shards = parallelism.context_parallel_degree * parallelism.tensor_parallel_degree
    if num_tokens <= 0 or num_tokens % shards:
        raise ValueError("Microbatch token count must be positive and divisible by CP * TP")
    return num_tokens // shards


def validate_dispatcher_capacity(
    model: BaseModel.Config, *, training: TrainingConfig, parallelism: ParallelismConfig
) -> None:
    """Validate the shared capacity contract of DeepEP and HybridEP configs."""
    for name, dispatcher, _, _ in model.traverse(BaseEPTokenDispatcher.Config):
        if isinstance(dispatcher, (DeepEPTokenDispatcher.Config, HybridEPTokenDispatcher.Config)):
            if parallelism.expert_parallel_degree <= 1:
                raise ValueError("DeepEP/HybridEP token dispatch requires expert_parallel_degree > 1")
            required = dispatcher_capacity(
                num_tokens=training.num_tokens_per_microbatch_per_dp_rank, parallelism=parallelism
            )
            if dispatcher.num_max_tokens_per_rank is None or dispatcher.num_max_tokens_per_rank < required:
                raise ValueError(f"{name}.num_max_tokens_per_rank must be at least {required}")
