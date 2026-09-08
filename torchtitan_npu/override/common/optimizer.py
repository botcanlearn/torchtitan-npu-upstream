# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overrides for NPU swap-backed optimizer states."""

from dataclasses import dataclass

import torch
import torch_npu
from torch.distributed._tensor import DTensor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import derive, override


def _make_swap(t: torch.Tensor) -> torch.Tensor:
    local = t.to_local() if isinstance(t, DTensor) else t
    # A DTensor may have p.numel() > 0 globally but local.numel() == 0 on ranks
    # that own an empty shard. Swap-memory allocation rejects zero-sized tensors,
    # so preserve such local shards with a regular empty tensor instead.
    out = (
        torch.empty_like(local)
        if local.numel() == 0
        else torch_npu.empty_with_swapped_memory(local.size(), dtype=local.dtype, device=local.device)
    )
    if isinstance(t, DTensor):
        return DTensor.from_local(
            out,
            t.device_mesh,
            t.placements,
            shape=t.size(),
            stride=t.stride(),
            run_check=False,
        )
    return out


def _swap_state_init_hook(optimizer, args, kwargs):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=p.device,
                )
                state["exp_avg"] = _make_swap(p).zero_()
                state["exp_avg_sq"] = _make_swap(p).zero_()


class VirtualOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts):
        super().__init__(config=config, model_parts=model_parts)
        for opt in self.optimizers:
            opt.register_step_pre_hook(_swap_state_init_hook)


@override(
    target=OptimizersContainer.Config,
    description="Allocate Adam/AdamW states in swap memory (host-offload)",
)
def virtual(
    cfg: OptimizersContainer.Config,
) -> VirtualOptimizersContainer.Config:
    return derive(cfg, VirtualOptimizersContainer.Config)
