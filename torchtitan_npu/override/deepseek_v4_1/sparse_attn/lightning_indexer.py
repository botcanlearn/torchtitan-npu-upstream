# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LI selection with a teacher-carrying autograd edge to SLIKG."""

from dataclasses import dataclass

import torch
from cann_ops_transformer import (
    lightning_indexer_metadata,
    sparse_lightning_indexer_kl_loss_grad_metadata,
)
from torchtitan.config import derive

from torchtitan_npu.models.deepseek_v4_1.indexer import ScoreAndSelect

_LAYOUT = "TND"
_MASK_MODE = 3  # right-down causal: entry j is complete for query t iff j < (t+1)//ratio


class _LightningIndexerTND(torch.autograd.Function):
    """LI v2 forward fused with the SLIKG backward, in the TND varlen layout.

    The kernels' contract (A3-verified): ``q``/``k`` are BF16 with a single
    key head and ``index_head_dim`` 128, ``w`` is float32 ``[T, Hi]``, the
    indices are document-local with ``-1`` padding, and the SLIKG metadata
    requires ``topk`` to be 512 or an integer multiple of 1024.  The forward
    returns the document-local indices (for SLIKG) and the scores; the caller
    globalizes the indices for the model contract.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        idx_q,
        idx_k,
        idx_w,
        topk,
        cu_seqlens_q,
        cu_seqlens_k,
        cmp_residual_k,
        li_metadata,
        slig_metadata,
        cmp_ratio,
    ):
        topk_local, topk_scores = torch.ops.cann_ops_transformer.lightning_indexer(
            idx_q,
            idx_k,
            idx_w,
            topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=li_metadata,
            layout_q=_LAYOUT,
            layout_k=_LAYOUT,
            mask_mode=_MASK_MODE,
            cmp_ratio=cmp_ratio,
            return_value=1,
        )
        ctx.save_for_backward(
            idx_q, idx_k, idx_w, topk_local, cu_seqlens_q, cu_seqlens_k, cmp_residual_k, slig_metadata
        )
        ctx.cmp_ratio = cmp_ratio
        return topk_local, topk_scores

    @staticmethod
    def backward(ctx, grad_topk_local, grad_topk_scores):  # pyrefly: ignore [bad-override]
        # The indices are integer selections: never tracked, always None.  The
        # scores' gradient is the teacher the SLIKG kernel expects.  A teacher
        # is a nonnegative attention mass; negative entries mean the loss side
        # was not swapped (the eager log-softmax student term produces
        # ``Z * Y - p``), which would silently corrupt the indexer gradients.
        if grad_topk_scores is None:
            return None, None, None, None, None, None, None, None, None, None
        (
            idx_q,
            idx_k,
            idx_w,
            topk_local,
            cu_seqlens_q,
            cu_seqlens_k,
            cmp_residual_k,
            slig_metadata,
        ) = ctx.saved_tensors
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_local,
            attn_softmax_l1_norm=grad_topk_scores,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=slig_metadata,
            layout_q=_LAYOUT,
            layout_k=_LAYOUT,
            mask_mode=_MASK_MODE,
            cmp_ratio=ctx.cmp_ratio,
        )
        # ``idx_w`` arrives float32 (the kernel contract); the cast back to the
        # projection dtype happens in the eager graph outside the Function.
        return dq, dk, dw, None, None, None, None, None, None, None


class AscScoreAndSelect(ScoreAndSelect):
    """The fused LI/SLIKG score-and-select for the V4.1 indexer layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(ScoreAndSelect.Config):
        @property
        def score_gradient(self) -> str:
            # The node always runs the LI/SLIKG kernel -- pool-configured
            # layers included, their candidate pool bypassed -- so every
            # layer's loss must hand the teacher to the fused backward.
            return "teacher"

    def _score_and_select(
        self,
        idx_q_BLHiDi: torch.Tensor,
        idx_k_BNDi: torch.Tensor,
        weights_BLHi: torch.Tensor,
        attention_masks,
        *,
        candidates_BLN: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cu_seqlens_q = attention_masks.cu_seq_q
        if cu_seqlens_q is None:
            raise ValueError("The fused LightningIndexer requires the packed document boundaries (cu_seq_q).")
        if idx_q_BLHiDi.shape[0] != 1:
            raise ValueError(
                f"The fused LightningIndexer requires the packed B=1 layout, got batch {idx_q_BLHiDi.shape[0]}."
            )

        ratio = self.compress_ratio
        topk = self.index_topk
        cu_seqlens_k = (cu_seqlens_q // ratio).to(torch.int32)
        # The true per-document remainder; the aligned loader always produces
        # zeros; the model requires every document to align to this ratio.
        cmp_residual_k = ((cu_seqlens_q[1:] - cu_seqlens_q[:-1]) % ratio).to(torch.int32) if ratio > 1 else None
        num_heads, head_dim = idx_q_BLHiDi.shape[2], idx_q_BLHiDi.shape[3]
        li_metadata = lightning_indexer_metadata(
            num_heads,
            1,
            head_dim,
            topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            layout_q=_LAYOUT,
            layout_k=_LAYOUT,
            mask_mode=_MASK_MODE,
            cmp_ratio=ratio,
        )
        slig_metadata = sparse_lightning_indexer_kl_loss_grad_metadata(
            num_heads,
            1,
            head_dim,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            topk=topk,
            layout_q=_LAYOUT,
            layout_k=_LAYOUT,
            mask_mode=_MASK_MODE,
            cmp_ratio=ratio,
        )

        topk_local, topk_scores = _LightningIndexerTND.apply(
            idx_q_BLHiDi.flatten(0, 1),
            idx_k_BNDi.flatten(0, 1).unsqueeze(1).contiguous(),
            weights_BLHi.flatten(0, 1).float().contiguous(),
            topk,
            cu_seqlens_q,
            cu_seqlens_k,
            cmp_residual_k,
            li_metadata,
            slig_metadata,
            ratio,
        )
        # The kernels speak document-local coordinates; the model's pool is
        # global.  Invalid slots stay -1 and valid ones shift by the query's
        # document start, so the consumer contract matches the eager path.
        starts = cu_seqlens_k.to(torch.long)[attention_masks.doc_ids_BL.reshape(-1)].view(-1, 1, 1)
        topk_indices_BLK = (
            torch.where(  # pyrefly: ignore [no-matching-overload]
                topk_local >= 0, topk_local + starts, topk_local
            )
            .reshape(1, -1, topk)
            .to(torch.long)
        )
        topk_scores_BLK = topk_scores.reshape(1, -1, topk)
        # The kernel's topk is pinned to the SLIKG metadata grid (512 or a
        # multiple of 1024), but the model contract clamps K to the pool size
        # exactly as the eager path does (``min(index_topk, N)``) — the eager
        # attention rejects a wider ``kv_indices``.  Compacting the invalid
        # slots to the back (the fused attention's pattern) and slicing to the
        # clamp keeps every selected entry; the sliced-away slots were padding,
        # so the backward still receives a zero gradient for them.
        k_eff = min(topk, idx_k_BNDi.shape[1])
        if k_eff < topk:
            invalid = topk_indices_BLK < 0
            order = invalid.to(torch.int32).argsort(dim=-1, stable=True)  # pyrefly: ignore [missing-attribute]
            topk_indices_BLK = topk_indices_BLK.gather(-1, order)[..., :k_eff]
            topk_scores_BLK = topk_scores_BLK.gather(-1, order)[..., :k_eff]
        return topk_indices_BLK, topk_scores_BLK, candidates_BLN


def derive_fused_score_and_select_config(cfg: ScoreAndSelect.Config) -> AscScoreAndSelect.Config:
    """Derive the fused score-and-select node; the indexer's norm/rope stay
    owned by the plain indexer and the blanket overrides cover them."""
    return derive(cfg, AscScoreAndSelect.Config)
