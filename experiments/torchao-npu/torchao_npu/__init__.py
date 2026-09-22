# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchao_npu.configs import (
    ModuleSwapConfig,
    ParamSwapConfig,
    QuantCompressorConfig,
    QuantLightningIndexerConfig,
    QuantV41SparseAttentionConfig,
)

__all__ = [
    "ModuleSwapConfig",
    "ParamSwapConfig",
    "QuantCompressorConfig",
    "QuantLightningIndexerConfig",
    "QuantV41SparseAttentionConfig",
]


def normalize_dim(dim: int, ndim: int) -> int:
    """Return ``dim`` as a non-negative index into an ``ndim``-D tensor."""
    if not -ndim <= dim < ndim:
        raise IndexError(f"dim {dim} out of range for {ndim}D tensor")
    return dim % ndim
