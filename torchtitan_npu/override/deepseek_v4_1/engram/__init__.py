# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 Engram Host table and sparse optimizer overrides."""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.engram_host import HostEngramTable

if TYPE_CHECKING:
    from .ascendc import HostOffloadEngramTable
    from .mxfp8 import MXFP8HostOffloadEngramTable


@override(
    target=HostEngramTable.Config,
    exact=True,
    description=(
        "Keep the EP-local Engram table and SparseAdam state on Host and use CANN EngramFetch/EngramFetchGrad"
    ),
)
def host_offload(
    cfg: HostEngramTable.Config,
    num_max_tokens_per_rank: int,
    pin_memory: bool = True,
) -> "HostOffloadEngramTable.Config":
    """Replace Torch Host lookup with CANN while retaining sparse training.

    Requires ``EP>1`` and a package with training-mode direct registration.
    The pinned CPU shard is registered once and updated in place by SparseAdam.
    ``num_max_tokens_per_rank`` counts flattened hash-row requests across every
    token and N-gram head, and must cover the largest batch used by the run.
    """
    from .ascendc import HostOffloadEngramTable

    return derive(
        cfg,
        HostOffloadEngramTable.Config,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        pin_memory=pin_memory,
    )


@override(
    target=HostEngramTable.Config,
    exact=True,
    description=(
        "Keep the authoritative Engram table on Host and use CANN EngramFetch with MXFP8 data and E8M0 scales"
    ),
)
def host_offload_mxfp8(
    cfg: HostEngramTable.Config,
    num_max_tokens_per_rank: int,
    pin_memory: bool = True,
    quantization_chunk_rows: int = 32768,
) -> "MXFP8HostOffloadEngramTable.Config":
    """Fetch MXFP8 rows while retaining FP32 SparseAdam master weights."""
    from .mxfp8 import MXFP8HostOffloadEngramTable

    return derive(
        cfg,
        MXFP8HostOffloadEngramTable.Config,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        pin_memory=pin_memory,
        quantization_chunk_rows=quantization_chunk_rows,
    )


__all__ = ["host_offload", "host_offload_mxfp8"]
