# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Patch ``create_varlen_metadata_for_document`` to pre-compute CPU cu_seq.

FA v3 ``sparse_mode=7`` requires ``actual_seq_qlen`` / ``actual_seq_kvlen``
on CPU.  Upstream ``create_varlen_metadata_for_document`` builds them on
the input device.  This patch moves the D2H transfer into the metadata
factory so that ``VarlenMetadata.cu_seq_q`` / ``cu_seq_k`` are already
CPU int64 when they enter the model forward.

The forward method still has a device-guard fallback for the case where
AC checkpoint or FSDP2 internals recreate the ``VarlenMetadata`` with
device tensors — but in the common case the patch eliminates the per-layer
D2H sync entirely.
"""

from __future__ import annotations

import torch
import torchtitan.models.common.attention as _titan_attn

# The decoder module imports create_varlen_metadata_for_document by name
# (``from torchtitan.models.common.attention import ...``), so replacing
# the attribute on ``attention`` alone is not enough — we must also update
# every module that already captured the original function reference.
import torchtitan.models.common.decoder as _titan_decoder
from torchtitan.tools.logging import logger

_orig_create_varlen = _titan_attn.create_varlen_metadata_for_document


def _patched_create_varlen(input_batch: torch.Tensor, eos_id: int):
    metadata = _orig_create_varlen(input_batch, eos_id)
    cpu_cu_seq_q = metadata.cu_seq_q.to(torch.int64).cpu()
    cpu_cu_seq_k = metadata.cu_seq_k.to(torch.int64).cpu()
    return metadata._replace(cu_seq_q=cpu_cu_seq_q, cu_seq_k=cpu_cu_seq_k)


_titan_attn.create_varlen_metadata_for_document = _patched_create_varlen
_titan_decoder.create_varlen_metadata_for_document = _patched_create_varlen
logger.info("[Patch] Patched create_varlen_metadata_for_document for CPU cu_seq")
