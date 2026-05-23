# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan/components/optimizer.py

Target:
- torchtitan.components.optimizer.register_moe_load_balancing_hook

Reason:
- DeepSeek-v4 hash MoE layers intentionally remove `expert_bias`, while upstream
  pre-hook unconditionally updates `moe.expert_bias`, which can raise
  AttributeError at runtime.
"""

import torch
import torch.distributed.tensor  # noqa: F401 - explicit import needed for isinstance check
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.tensor import Replicate

from torchtitan.components import optimizer as tt_optimizer
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


_ORIG_REGISTER_MOE_LOAD_BALANCING_HOOK = getattr(
    tt_optimizer,
    "register_moe_load_balancing_hook",
    None,
)


def _get_layers(model_part: nn.Module) -> nn.ModuleDict:
    layers = model_part.get_submodule("layers")
    if not isinstance(layers, nn.ModuleDict):
        raise TypeError(
            f"Expected model_part.layers to be nn.ModuleDict, "
            f"got {type(layers).__name__}"
        )
    return layers


def register_moe_load_balancing_hook_with_expert_bias_guard(
    optimizers: tt_optimizer.OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            layers = _get_layers(model_part)
            for transformer_block in layers.values():
                if transformer_block.moe_enabled:
                    # pyrefly: ignore [missing-attribute]
                    return bool(transformer_block.moe.load_balance_coeff)
        return False

    def _is_recomputation_enabled(module: nn.Module) -> bool:
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ) -> None:
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        tokens_per_expert_list: list[torch.Tensor] = []

        for model_part in model_parts:
            layers = _get_layers(model_part)
            for transformer_block in layers.values():
                if not transformer_block.moe_enabled:
                    continue
                # pyrefly: ignore [missing-attribute]
                if transformer_block.moe.load_balance_coeff is None:
                    return
                # pyrefly: ignore [missing-attribute]
                tokens_per_expert = transformer_block.moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_list.append(tokens_per_expert)

        tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

        if loss_mesh is not None:
            if isinstance(tokens_per_expert_by_layer, torch.distributed.tensor.DTensor):
                tokens_per_expert_by_layer = tokens_per_expert_by_layer.redistribute(
                    placements=[Replicate()]
                    * tokens_per_expert_by_layer.device_mesh.ndim
                )
            else:
                pg = loss_mesh.get_group()
                torch.distributed.all_reduce(
                    tokens_per_expert_by_layer,
                    group=pg,
                    op=torch.distributed.ReduceOp.SUM,
                )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                layers = _get_layers(model_part)
                for transformer_block in layers.values():
                    if not transformer_block.moe_enabled:
                        continue
                    moe = transformer_block.moe

                    tokens_per_expert = tokens_per_expert_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # pyrefly: ignore [missing-attribute]
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        tokens_per_expert.mean() - tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    expert_bias = getattr(moe, "expert_bias", None)
                    if expert_bias is not None:
                        expert_bias.add_(expert_bias_delta)
                    # pyrefly: ignore [missing-attribute]
                    moe.tokens_per_expert.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts,
                parallel_dims=parallel_dims,
            )
        )


if _ORIG_REGISTER_MOE_LOAD_BALANCING_HOOK is not None:
    tt_optimizer.register_moe_load_balancing_hook = (
        register_moe_load_balancing_hook_with_expert_bias_guard
    )
else:
    logger.warning(
        "[Optimizer Patch] register_moe_load_balancing_hook not found in upstream; skip patching"
    )
