# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU VarlenAttention using CANN FA v3 with sparse_mode=7.

Extends upstream VarlenAttention via the inner_attention plugin mechanism.
CP uses Ulysses-style all-to-all (NPUVarlenUlyssesCP) — each rank sees
full sequence with partial heads, document boundaries are correctly detected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata

_MASK_SIZE = 2048
_SPARSE_MODE_VARLEN = 7  # per-document causal (varlen) in CANN FA v3


class NPUVarlenAttention(VarlenAttention):
    """Per-document causal attention via CANN FA v3 in TND layout."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        from torchtitan_npu.distributed.context_parallel.npu_varlen_cp import (
            ensure_mask_handler_registered,
        )

        ensure_mask_handler_registered()

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Per-document causal attention via FA v3.
        CP support: transparent — ``NPUVarlenUlyssesCP`` (registered in
        ``npu_varlen_cp.py``) applies all-to-all pre/post hooks, so this
        forward always sees the full sequence regardless of CP degree.
        """
        bs, seqlen, n_local, head_dim = xq.shape
        # BSND → TND layout
        total_tokens = bs * seqlen
        q_tnd = xq.reshape(total_tokens, n_local, head_dim)
        k_tnd = xk.reshape(total_tokens, -1, head_dim)
        v_tnd = xv.reshape(total_tokens, -1, head_dim)
        input_dtype = q_tnd.dtype
        q_tnd = q_tnd.to(torch.bfloat16)
        k_tnd = k_tnd.to(torch.bfloat16)
        v_tnd = v_tnd.to(torch.bfloat16)

        # FA v3 requires actual_seq_qlen on CPU; the patched
        # create_varlen_metadata_for_document already moved cu_seq_q to
        # CPU, but AC checkpoint may recreate VarlenMetadata with device
        # tensors.  Guard: if already CPU, use directly; otherwise D2H.
        cu_seq = attention_masks.cu_seq_q
        cu_seq = cu_seq.to(torch.int64).cpu() if cu_seq.device.type != "cpu" else cu_seq.to(torch.int64)
        cu_seq = cu_seq[1:]
        _scale = scale if scale is not None else head_dim**-0.5

        if not hasattr(self, "_causal_mask"):
            self._causal_mask = torch.triu(
                torch.ones(_MASK_SIZE, _MASK_SIZE, dtype=torch.bool, device=q_tnd.device),
                diagonal=1,
            )

        output_tnd = torch.ops.npu.npu_fusion_attention_v3(
            query=q_tnd,
            key=k_tnd,
            value=v_tnd,
            head_num=n_local,
            input_layout="TND",
            atten_mask=self._causal_mask,
            scale=_scale,
            keep_prob=1.0,
            pre_tockens=seqlen,
            next_tockens=0,
            actual_seq_qlen=cu_seq,
            actual_seq_kvlen=cu_seq,
            sparse_mode=_SPARSE_MODE_VARLEN,
        )[0]

        output = output_tnd.view(bs, seqlen, n_local, head_dim)
        return output.to(input_dtype)
