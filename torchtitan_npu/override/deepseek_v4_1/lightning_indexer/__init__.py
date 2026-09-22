# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 LightningIndexer override: selection fused with its SLIKG backward.

Independent of :mod:`~torchtitan_npu.override.deepseek_v4_1.sparse_attn` on purpose.  The
two are separate kernels with separate targets -- this one replaces the selection node,
that one the sparse core -- and a run may take either, both, or neither.  They must agree
on one coordinate system (both speak document-local indices), which is a contract rather
than a shared module.
"""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.indexer import Selector


@override(
    target=Selector.Config,
    exact=True,
    description=(
        "Fused LightningIndexer selection with its SLIKG backward in one "
        "autograd.Function (every indexer layer; the candidate pool is not part of "
        "the fused path)"
    ),
)
def asc(cfg: Selector.Config):
    from .ascendc import AscSelector

    return derive(cfg, AscSelector.Config)
