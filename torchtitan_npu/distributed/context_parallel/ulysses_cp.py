# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ulysses Context Parallel — ParallelStyle with pre/post hooks.

Pre-hook:  all_to_all on Q/K/V to swap heads ↔ sequence across CP ranks.
Post-hook: reverse all_to_all on the output.

No forward replacement — the original module's internal transpose works
correctly with the all-to-all'd tensor shapes.
"""

from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torchtitan.models.common.attention import ScaledDotProductAttention

from .registry import register_cp_strategy


class AllToAll(torch.autograd.Function):
    """All-to-all with scatter on one dim and gather on another.

    Forward:  ``chunk(scatter_dim)`` → ``dist.all_to_all`` → ``cat(gather_dim)``.
    Backward: reverse — chunk on the forward's gather_dim, all-to-all in
              reverse direction, cat on the forward's scatter_dim.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input_tensor, mesh, scatter_dim, gather_dim):
        ctx.mesh = mesh
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        world_size = mesh.size()
        input_list = [t.contiguous() for t in list(input_tensor.chunk(world_size, dim=scatter_dim))]
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
        dist.all_to_all(output_list, input_list, group=mesh.get_group())
        output = torch.cat(output_list, dim=gather_dim)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        world_size = ctx.mesh.size()

        grad_list = [t.contiguous() for t in list(grad_output.chunk(world_size, dim=ctx.gather_dim))]
        grad_output_list = [torch.empty_like(grad_list[0]) for _ in range(world_size)]
        dist.all_to_all(grad_output_list, grad_list, group=ctx.mesh.get_group())
        grad_input = torch.cat(grad_output_list, dim=ctx.scatter_dim)
        return grad_input, None, None, None


def all_to_all(input_tensor, mesh, scatter_dim, gather_dim):
    """Safe wrapper around ``AllToAll``."""
    return AllToAll.apply(input_tensor, mesh, scatter_dim, gather_dim)


class UlyssesCP(ParallelStyle):
    """Ulysses Context Parallel — all-to-all head/sequence swap.

    Applies to ``ScaledDotProductAttention`` modules.  The module's internal
    transpose between BSND ↔ BNSD works correctly with the all-to-all'd shapes.
    """

    @staticmethod
    def _pre_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: DeviceMesh,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        q = all_to_all(args[0], mesh, scatter_dim=2, gather_dim=1)
        k = all_to_all(args[1], mesh, scatter_dim=2, gather_dim=1)
        v = all_to_all(args[2], mesh, scatter_dim=2, gather_dim=1)
        return (q, k, v, *args[3:]), kwargs

    @staticmethod
    def _post_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        output: Any,
        mesh: DeviceMesh,
    ) -> Any:
        return all_to_all(output, mesh, scatter_dim=1, gather_dim=2)

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        if not isinstance(module, ScaledDotProductAttention):
            raise TypeError(f"UlyssesCP expects ScaledDotProductAttention, got {type(module).__name__}")
        module.register_forward_pre_hook(partial(self._pre_hook, mesh=device_mesh), with_kwargs=True)
        module.register_forward_hook(partial(self._post_hook, mesh=device_mesh))
        return module


def _detect_ulysses(module: torch.nn.Module) -> bool:
    return isinstance(module, ScaledDotProductAttention)


def _apply_ulysses(module: torch.nn.Module, cp_mesh: DeviceMesh) -> None:
    parallelize_module(module, cp_mesh, UlyssesCP())


register_cp_strategy(_detect_ulysses, _apply_ulysses)
