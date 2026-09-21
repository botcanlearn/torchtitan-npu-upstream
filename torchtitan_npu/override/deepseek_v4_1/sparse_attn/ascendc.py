# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""V4.1 SMLA attention and SMLAG-backed indexer distillation."""

from dataclasses import dataclass

import torch
from cann_ops_transformer import sparse_flash_mla_grad_metadata, sparse_flash_mla_metadata

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.override import _IS_A5


def _kernel_options(cu_seqlens_q, cu_seqlens_cmp_kv, cmp_residual_kv, ratio, window_size):
    return dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_q,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        cmp_ratio=max(ratio, 1),
        ori_mask_mode=4,
        cmp_mask_mode=0 if _IS_A5 and cu_seqlens_cmp_kv is None else 3,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
    )


class _SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        swa_k,
        cmp_k,
        cmp_sparse_indices,
        sinks,
        cu_seqlens_q,
        cu_seqlens_cmp_kv,
        cmp_residual_kv,
        softmax_scale,
        ratio,
        window_size,
        topk_scores,
        teacher_scale,
        query_valid,
        score_gradient,
        aux_loss,
    ):
        options = _kernel_options(cu_seqlens_q, cu_seqlens_cmp_kv, cmp_residual_kv, ratio, window_size)
        # K can differ between the candidate and indexer paths. Use the actual
        # supplied shape rather than assuming the LI configuration's top-k.
        geometry = dict(
            ori_topk=0,
            cmp_topk=0 if cmp_sparse_indices is None else cmp_sparse_indices.shape[-1],
            has_ori_kv=True,
            has_cmp_kv=cmp_k is not None,
        )
        smla_metadata = sparse_flash_mla_metadata(
            q.shape[1],
            1,
            q.shape[2],
            ori_topk_length=None,
            cmp_topk_length=None,
            **options,
            **geometry,
        )
        smla_grad_metadata = sparse_flash_mla_grad_metadata(q.shape[1], 1, q.shape[2], **options, **geometry)
        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sinks,
            metadata=smla_metadata,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            sinks,
            cu_seqlens_q,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            output,
            lse,
            smla_grad_metadata,
            topk_scores,
            query_valid,
        )
        ctx.softmax_scale, ctx.ratio, ctx.window_size = softmax_scale, ratio, window_size
        # The teacher consumes the LSE as a detached constant.
        ctx.mark_non_differentiable(lse)
        ctx.teacher_scale = teacher_scale
        ctx.score_gradient = score_gradient
        ctx.aux_loss = aux_loss
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        del grad_lse  # the LSE is a teacher signal, never a loss input
        # A single read of the saved tensors: eager AC recomputation must see
        # the same native metadata the forward produced.
        (
            q,
            swa_k,
            cmp_k,
            cmp_sparse_indices,
            sinks,
            cu_seqlens_q,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            output,
            lse,
            smla_grad_metadata,
            topk_scores_saved,
            query_valid,
        ) = ctx.saved_tensors
        dq, dswa_k, dcmp_k, dsinks, _, cmp_softmax_l1_norm = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            output,
            lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            ori_sparse_indices=None,
            cmp_sparse_indices=cmp_sparse_indices,
            sinks=sinks,
            metadata=smla_grad_metadata,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            ori_topk_length=None,
            cmp_topk_length=None,
            softmax_scale=ctx.softmax_scale,
            **_kernel_options(cu_seqlens_q, cu_seqlens_cmp_kv, cmp_residual_kv, ctx.ratio, ctx.window_size),
        )
        grad_topk_scores = None
        if topk_scores_saved is not None:
            valid = cmp_sparse_indices >= 0
            if query_valid is not None:
                valid = valid & query_valid.reshape(-1, 1, 1)
            p = cmp_softmax_l1_norm.reshape_as(topk_scores_saved).float().masked_fill(~valid, 0.0)
            if ctx.score_gradient == "teacher":
                grad_topk_scores = p
            else:
                logits = topk_scores_saved.float().masked_fill(~valid, -torch.inf)
                logits = logits.masked_fill(~valid.any(-1, keepdim=True), 0.0)
                student = logits.softmax(-1).masked_fill(~valid, 0.0)
                grad_topk_scores = student * p.sum(-1, keepdim=True) - p
            grad_topk_scores = (grad_topk_scores * ctx.teacher_scale).to(topk_scores_saved.dtype)
            ctx.aux_loss.record_teacher(topk_scores_saved, p, valid)
        return (
            dq,
            dswa_k,
            dcmp_k if cmp_k is not None else None,
            None,
            dsinks,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_topk_scores,
            None,
            None,
            None,
            None,
        )


def _localize_indices(
    topk_indices: torch.Tensor, doc_ids_BL: torch.Tensor, cu_seqlens_cmp_kv: torch.Tensor
) -> torch.Tensor:
    """Global compressed-pool coordinates into the query's own document-local grid.

    Entries outside the query's document become ``-1``; the caller compacts them.
    The shared ``topk_indices``/``topk_scores`` are never modified in place.
    """
    starts = cu_seqlens_cmp_kv.to(dtype=torch.long, device=topk_indices.device)
    doc_ids = doc_ids_BL.reshape(-1)
    doc_start = starts[doc_ids].unsqueeze(-1)
    doc_end = starts[doc_ids + 1].unsqueeze(-1)
    indices = topk_indices.reshape(-1, topk_indices.shape[-1]).to(torch.long)
    valid = (indices >= doc_start) & (indices < doc_end)
    return torch.where(valid, indices - doc_start, -1).view_as(topk_indices)


class AscV41SparseAttention(CompressedSparseInnerAttention2):
    """The A5 TND kernel behind the reference forward contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention2.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        cmp_k=None,
        *,
        attention_masks,
        topk_indices=None,
        topk_scores=None,
        attn_sink=None,
    ):
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together")
        wants_teacher = self.training and self.aux_loss is not None and cmp_k is not None
        metadata = attention_masks
        ratio = self.compress_ratio
        if ratio not in (0, 1, 2):
            raise ValueError(f"V4.1 fused sparse attention supports ratios 0/1/2, got {ratio}")
        if q.ndim != 4 or q.shape[0] != 1 or swa_k.shape != (1, q.shape[1], q.shape[-1]):
            raise ValueError("V4.1 SMLA requires CP1 packed Q [1,S,H,D] and window KV [1,S,D]")
        if attn_sink is None:
            raise ValueError("V4.1 SMLA requires per-head attention sinks")
        if q.dtype != torch.bfloat16 or swa_k.dtype != q.dtype:
            raise ValueError("V4.1 SMLA requires BF16 Q/KV; no implicit precision conversion")
        cu_seqlens_q = metadata.cu_seq_q
        if cu_seqlens_q is None:
            raise ValueError("V4.1 SMLA requires the packed document boundaries (cu_seq_q metadata)")
        cmp_k_tnd = cu_seqlens_cmp_kv = cmp_residual_kv = cmp_sparse_indices = None
        carrier = teacher_scale = None
        if ratio == 0:
            if cmp_k is not None:
                raise ValueError("window-only ratio 0 must not receive a second KV stream")
        else:
            if cmp_k is None or cmp_k.ndim != 3 or cmp_k.shape[0] != 1 or cmp_k.dtype != q.dtype:
                raise ValueError("ratio 1/2 requires a BF16 second KV stream [1,N,D]")
            if topk_indices is None:
                raise ValueError("ratio 1/2 fused attention requires the model's selection indices")
            if ratio == 1 and cmp_k.shape != swa_k.shape:
                raise ValueError("ratio 1 requires the full-resolution second KV stream")
            # Per-document alignment makes every document's compressed length
            # exactly cu_seqlens_q // ratio.  Like DSV4, the uncompressed
            # path carries no residual; unlike DSV4, ratio 1 keeps its
            # full-resolution second KV stream (the model's difference from
            # DSV4's has_compressed gate) — only the residual is None.
            cu_seqlens_cmp_kv = cu_seqlens_q // ratio
            cmp_residual_kv = torch.zeros_like(cu_seqlens_cmp_kv[1:]) if ratio > 1 else None
            cmp_k_tnd = cmp_k.flatten(0, 1)
            # TND kernels consume document-local indices; cu_seqlens_cmp_kv
            # supplies the document offsets. Preserve the reference
            # cross-document mask.
            local = _localize_indices(topk_indices, metadata.doc_ids_BL, cu_seqlens_cmp_kv)
            # Packed documents can leave leading/interleaved invalid slots;
            # compact them without changing the selected keys or their order.
            order = (local < 0).to(torch.int32).argsort(dim=-1, stable=True)
            cmp_sparse_indices = local.gather(-1, order).flatten(0, 1).to(torch.int32).unsqueeze(1).contiguous()
            cmp_k_tnd = cmp_k_tnd.unsqueeze(1).contiguous()
            if wants_teacher:
                assert self.aux_loss is not None
                if topk_scores is None:
                    raise ValueError("Indexer distillation requires topk_scores")
                carrier = topk_scores.gather(-1, order).flatten(0, 1).unsqueeze(1).contiguous()
                teacher_scale = self.aux_loss.teacher_alpha()
        output, _lse = _SparseMLA.apply(
            q.flatten(0, 1).contiguous(),
            swa_k.flatten(0, 1).unsqueeze(1).contiguous(),
            cmp_k_tnd,
            cmp_sparse_indices,
            attn_sink.float(),
            cu_seqlens_q,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            self.softmax_scale,
            ratio,
            self.window_size,
            carrier,
            teacher_scale,
            attention_masks.valid_tokens_BL,
            self.score_gradient,
            self.aux_loss if wants_teacher else None,
        )
        return output.reshape_as(q)
