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

The replacement drives the quantized ``ds41`` pair rather than the plain bf16 selection:
``quant_lightning_indexer`` on the Full Mode layer that builds the candidate pool or on a
layer outside the hierarchy, and ``quant_sparse_lightning_indexer`` on a layer that searches
it.  Which kernel runs follows from ``mode`` together with the pool capacity -- the same pair
the reference selector dispatches on -- so the candidate mechanism crosses the override
boundary with the selector instead of being bypassed.

The entry takes one switch, and it **defaults to the pre-quantization kernel**: the
quantized path currently dies on the second training step with an aicore timeout, so the
safe kernel is the one a run gets unless it asks for the other.  A run that wants the
candidate pool has to say so explicitly:

    --override.imports \
      'torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc={"legacy":false}'
"""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.indexer import Selector

if TYPE_CHECKING:
    from .ascendc import AscSelector


@override(
    target=Selector.Config,
    exact=True,
    description=(
        "LightningIndexer selection fused with its SLIKG backward in one autograd.Function "
        "(every indexer layer); legacy=true, the default, selects the pre-quantization "
        "kernel and legacy=false the ds41 QLI/QSLI candidate pair"
    ),
)
def asc(cfg: Selector.Config, legacy: bool = True) -> "AscSelector.Config":
    from .ascendc import AscSelector

    return derive(cfg, AscSelector.Config, legacy=legacy)
