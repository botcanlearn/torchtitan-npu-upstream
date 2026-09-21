# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Drop int64 from the AscendC gather-family lowering whitelist.

torch_npu's ``gather_lowering`` registers ``aten.gather`` /
``aten.embedding`` / ``aten.index_select`` with int64 in the supported
dtype list.  The fused IndirectLoad path then fails AscendC dtype
inference for int64 index tensors ("Infer dtype failed for
graph_hint/indirectload IndirectLoad; input_dtypes: [DT_INT64, DT_INT64]
is not supported now"), which aborts whole-graph inductor compilation.
Re-register the guard with int32 only so int64 gather-family nodes take
the eager fallback instead of the fused AscendC lowering.  The SoC gate
of the original registration is intentionally not mirrored: the narrowed
entry applies on every SoC.
"""

from torchtitan.tools.logging import logger


def apply() -> None:
    try:
        import torch

        # Force the original registration first: it runs at module import
        # (otherwise lazily at compile time), and that later import would
        # overwrite this patch's narrower entry.
        import torch_npu._inductor.ascendc.lowering.gather_lowering  # noqa: F401  # pyrefly: ignore [missing-import]
        from torch_npu._inductor.ascendc.lowering.common import _LoweringGuard, float_dtypes
    except Exception as exc:  # torch_npu variants differ widely
        logger.warning("[PATCH] ascendc gather dtype restriction skipped: %s", exc)
        return

    aten = torch.ops.aten
    dtypes = (*float_dtypes(), torch.int32)
    for op in (aten.gather, aten.embedding, aten.index_select):
        _LoweringGuard.support(op, dtypes)  # pyrefly: ignore [bad-argument-type]
    logger.info(
        "[PATCH] ascendc gather/embedding/index_select lowering dtypes narrowed to %s (int64 falls back to eager)",
        dtypes,
    )


apply()
