# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from PyTorch,
# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/pipelining/stage.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import torch
import torch.distributed.pipelining.stage

try:
    import torchtitan.trainer as titan_trainer
except ImportError:
    import torchtitan.train as titan_trainer
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining._backward import (
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from torch.nn.parallel import DistributedDataParallel
from torchtitan.tools.logging import logger

from torchtitan_npu.models.deepseek_v4.pipeline_parallel import (
    _is_deepseek_v4_pp_target,
    _with_deepseek_v4_pp_input_ids,
)


def _patch_fork_rng_for_npu_pipeline() -> None:
    if getattr(torch.random.fork_rng, "npu_pipeline_rng_patched", False):
        return

    original_fork_rng = torch.random.fork_rng

    @wraps(original_fork_rng)
    def fork_rng_with_npu_device_type(devices=None, *args, **kwargs):
        # PyTorch pipeline internals call fork_rng with pipeline stage devices but
        # do not pass device_type, which defaults to CUDA upstream. Keep explicit
        # device_type calls untouched, and only redirect device-scoped calls used
        # by pipeline execution to NPU.
        if devices is not None and "device_type" not in kwargs:
            kwargs["device_type"] = "npu"
        return original_fork_rng(devices, *args, **kwargs)

    vars(fork_rng_with_npu_device_type)["npu_pipeline_rng_patched"] = True
    torch.random.fork_rng = fork_rng_with_npu_device_type
    logger.info("[Patch] Registered NPU fork_rng default device_type hook.")


_patch_fork_rng_for_npu_pipeline()


def _patch_post_dataloading_process_for_deepseek_v4_pp_input_ids() -> None:
    if getattr(titan_trainer.Trainer, "npu_dsv4_pp_input_ids_patched", False):
        return

    original = titan_trainer.Trainer.post_dataloading_process

    @wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _is_deepseek_v4_pp_target(self):
            result = _with_deepseek_v4_pp_input_ids(self, result)
        return result

    titan_trainer.Trainer.post_dataloading_process = wrapper
    titan_trainer.Trainer.npu_dsv4_pp_input_ids_patched = True
    logger.info("[Patch] Registered DeepSeekV4 PP input_ids kwargs forwarding hook.")


_patch_post_dataloading_process_for_deepseek_v4_pp_input_ids()


def backward_maybe_with_nosync(
    self,
    backward_type,
    bwd_kwargs: dict,
    last_backward: bool = False,
) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None]:
    """
    Whether using PP with FSDP, DDP, or replicate there are some runtime differences between the last backward step and the
    other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
    there are additional state-variables and performance considerations depending on the data parallelism used.
    This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
    """

    def perform_backward(
        backward_type,
    ) -> Callable[
        [],
        tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None],
    ]:
        if backward_type == "full":
            return lambda: (
                stage_backward(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                ),
                None,
            )
        elif backward_type == "input":
            return lambda: stage_backward_input(
                bwd_kwargs["stage_output"],
                bwd_kwargs["output_grads"],
                bwd_kwargs["input_values"],
                self.submod.parameters(),
            )
        elif backward_type == "weight":
            return lambda: (
                stage_backward_weight(
                    self.submod.parameters(), bwd_kwargs["param_groups"]
                ),
                None,
            )
        else:
            raise RuntimeError(f"Unknown backward type: {backward_type}")

    # If submod is wrapped by DDP
    if isinstance(self.submod, DistributedDataParallel):
        if last_backward:
            # Last chunk, prepare for gradient reduction
            # NOTE: reaching into DDP implementation details here. Is there a better way?
            self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                list(
                    torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                        bwd_kwargs["stage_output"]
                    )
                )
            )
            result = perform_backward(backward_type)()
        else:
            with self.submod.no_sync():  # type: ignore[operator]
                result = perform_backward(backward_type)()

    # If submod is a FSDP or replicate module
    elif isinstance(self.submod, FSDPModule):
        self.submod.set_is_last_backward(False)
        # NOTE: npu modification start
        self.submod.set_reshard_after_backward(
            True
        )  # set True to save memory by resharding params
        self.submod.set_requires_gradient_sync(
            True
        )  # set True to save memory by resharding grads
        # NOTE: npu modification end
        result = perform_backward(backward_type)()

    else:
        # Non-DP submodule, regular backward
        result = perform_backward(backward_type)()

    grads, param_groups = result
    return grads, param_groups


# apply patch to reshard params and grads after backward to save memory, but this will hurt efficiency
torch.distributed.pipelining.stage._PipelineStageBase.backward_maybe_with_nosync = (
    backward_maybe_with_nosync
)
