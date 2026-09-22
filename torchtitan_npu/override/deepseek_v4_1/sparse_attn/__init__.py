# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent V4.1 attention and score-and-select overrides."""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.indexer import ScoreAndSelect


@override(
    target=CompressedSparseInnerAttention2.Config,
    exact=True,
    description="V4.1 SMLA forward and backward with the full-softmax LSE",
)
def asc(cfg: CompressedSparseInnerAttention2.Config):
    from .ascendc import AscV41SparseAttention

    return derive(cfg, AscV41SparseAttention.Config)


@override(
    target=ScoreAndSelect.Config,
    exact=True,
    description=(
        "The score-and-select node fused with its SLIKG backward in one "
        "autograd.Function (every indexer layer; the candidate pool is not "
        "part of the fused path)"
    ),
)
def asc_li(cfg: ScoreAndSelect.Config):
    from .lightning_indexer import derive_fused_score_and_select_config

    return derive_fused_score_and_select_config(cfg)
