# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ulysses-style CP for NPUVarlenAttention — reuses ulysses_cp.all_to_all.

Pre-hook: all_to_all on Q/K/V.
Post-hook: reverse all_to_all on output.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.distributed.context_parallel.ulysses_cp import all_to_all
from torchtitan_npu.models.common.npu_varlen_attention import NPUVarlenAttention

from .registry import register_cp_mask_handler, register_cp_strategy

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.distributed.device_mesh import DeviceMesh


class NPUVarlenUlyssesCP(ParallelStyle):
    @staticmethod
    def _pre_hook(module, args, kwargs, mesh):
        q = all_to_all(args[0], mesh, scatter_dim=2, gather_dim=1)
        k = all_to_all(args[1], mesh, scatter_dim=2, gather_dim=1)
        v = all_to_all(args[2], mesh, scatter_dim=2, gather_dim=1)
        return (q, k, v, *args[3:]), kwargs

    @staticmethod
    def _post_hook(module, args, output, mesh):
        return all_to_all(output, mesh, scatter_dim=1, gather_dim=2)

    def _apply(self, module, device_mesh):
        if not isinstance(module, NPUVarlenAttention):
            raise TypeError(f"NPUVarlenUlyssesCP expects NPUVarlenAttention, got {type(module).__name__}")
        module.register_forward_pre_hook(partial(self._pre_hook, mesh=device_mesh), with_kwargs=True)
        module.register_forward_hook(partial(self._post_hook, mesh=device_mesh))
        return module


def _detect_npu_varlen(module: nn.Module) -> bool:
    return isinstance(module, NPUVarlenAttention)


def _apply_npu_varlen(module: nn.Module, cp_mesh: DeviceMesh) -> None:
    parallelize_module(module, cp_mesh, NPUVarlenUlyssesCP())


register_cp_strategy(_detect_npu_varlen, _apply_npu_varlen)

_mask_handler_registered = False


def _varlen_cp_mask_handler(attention_masks: Any, cp_mesh: DeviceMesh) -> VarlenMetadata | None:
    # Upstream cp_shard expects BlockMask/dict and will sequence-shard them.
    # VarlenMetadata (document boundaries) must NOT be sharded — each CP rank
    # needs the full metadata to correctly compute attention masks after the
    # all-to-all.  By returning the metadata unchanged (handled=True tells
    # cp_shard to skip the default sharding path), we short-circuit upstream
    # sharding and preserve the original metadata for all CP ranks.
    if isinstance(attention_masks, VarlenMetadata):
        return attention_masks
    return None


def ensure_mask_handler_registered() -> None:
    global _mask_handler_registered
    if not _mask_handler_registered:
        register_cp_mask_handler(_varlen_cp_mask_handler)
        _mask_handler_registered = True
