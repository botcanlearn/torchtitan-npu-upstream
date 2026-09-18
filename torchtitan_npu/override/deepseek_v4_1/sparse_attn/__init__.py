# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""V4.1 fused sparse attention; the distillation loss stays owned by the model."""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2


@override(
    target=CompressedSparseInnerAttention2.Config,
    exact=True,
    description="V4.1 SMLA forward and backward with the full-softmax LSE",
)
def asc(cfg: CompressedSparseInnerAttention2.Config):
    from .ascendc import AscV41SparseAttention

    return derive(cfg, AscV41SparseAttention.Config)
