# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""V4.1 sparse attention with FP8 SWA fake quantization.

Main KV arrives already fake-quantized by the source Compressor and is shared
by all consuming layers. Forward and backward use the same fake-quantized
main KV and SWA values. SWA gradients pass directly to the original input
with an identity STE.
"""

__all__ = ["QuantV41SparseAttention"]

import torch
import torch_npu
from cann_ops_transformer import sparse_flash_mla_grad_metadata, sparse_flash_mla_metadata

from torchao_npu.ops.kv_cache_fake_quant import fake_quantize_mx_bf16

try:
    _IS_A5 = torch_npu.npu.get_device_name().startswith("Ascend950")
except (AttributeError, TypeError):
    _IS_A5 = False


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

        qdq_swa_k = fake_quantize_mx_bf16(swa_k, quant_group_size=32, quant_mode="mxfp8_bf16")

        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=qdq_swa_k,
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
            qdq_swa_k,
            cmp_k,
            cmp_sparse_indices,
            sinks,
            cu_seqlens_q,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            output,
            lse,
            smla_grad_metadata,
        )
        ctx.softmax_scale, ctx.ratio, ctx.window_size = softmax_scale, ratio, window_size
        # The teacher consumes the LSE as a detached constant.
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        del grad_lse  # the LSE is a teacher signal, never a loss input
        # A single read of the saved tensors: eager AC recomputation must see
        # the same native metadata the forward produced.
        (
            q,
            qdq_swa_k,
            cmp_k,
            cmp_sparse_indices,
            sinks,
            cu_seqlens_q,
            cu_seqlens_cmp_kv,
            cmp_residual_kv,
            output,
            lse,
            smla_grad_metadata,
        ) = ctx.saved_tensors
        dq, dswa_k, dcmp_k, dsinks, _, _ = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            output,
            lse,
            ori_kv=qdq_swa_k,
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
        # SWA Q/DQ runs inside this custom forward; apply its identity STE here.
        return dq, dswa_k, dcmp_k if cmp_k is not None else None, None, dsinks, None, None, None, None, None, None


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


class QuantV41SparseAttention(torch.nn.Module):
    """Sparse computation installed at the host's ``_compute_attention`` seam.

    The host supplies window_size, softmax_scale and compress_ratio when the
    module-swap handler binds this forward method to it.
    """

    def __init__(self, *, window_size: int, softmax_scale: float, compress_ratio: int):
        super().__init__()
        self.window_size = window_size
        self.softmax_scale = softmax_scale
        self.compress_ratio = compress_ratio

    def forward(
        self,
        q,
        swa_k,
        cmp_k,
        *,
        attention_masks,
        topk_indices,
        attn_sink,
        wants_teacher,
    ):
        metadata = attention_masks
        ratio = self.compress_ratio
        if ratio not in (0, 1, 2):
            raise ValueError(f"V4.1 fused sparse attention supports ratios 0/1/2, got {ratio}")
        if q.ndim != 4 or q.shape[0] != 1 or swa_k.shape != (1, q.shape[1], q.shape[-1]):
            raise ValueError("V4.1 SMLA requires CP1 packed Q [1,S,H,D] and original KV [1,S,D]")
        if q.shape[-1] != 512:
            raise ValueError("mixed quant V4.1 SMLA requires head_dim=512 with 64 trailing RoPE channels")
        if attn_sink is None:
            raise ValueError("V4.1 SMLA requires per-head attention sinks")
        if q.dtype != torch.bfloat16 or swa_k.dtype != q.dtype:
            raise ValueError("V4.1 SMLA requires BF16 Q/KV; no implicit precision conversion")
        cu_seqlens_q = metadata.cu_seq_q
        if cu_seqlens_q is None:
            raise ValueError("V4.1 SMLA requires the packed document boundaries (cu_seq_q metadata)")
        cmp_k_tnd = cu_seqlens_cmp_kv = cmp_residual_kv = cmp_sparse_indices = None
        if ratio == 0:
            if cmp_k is not None:
                raise ValueError("window-only ratio 0 must not receive a second KV stream")
        else:
            if (
                cmp_k is None
                or cmp_k.ndim != 3
                or cmp_k.shape[0] != 1
                or cmp_k.shape[-1] != q.shape[-1]
                or cmp_k.dtype != q.dtype
            ):
                raise ValueError("ratio 1/2 requires a BF16 second KV stream [1,N,D]")
            if topk_indices is None:
                raise ValueError("ratio 1/2 fused attention requires the model's selection indices")
            if ratio == 1 and cmp_k.shape != swa_k.shape:
                raise ValueError("ratio 1 requires full-resolution shared KV")
            # Per-document alignment makes every document's compressed length
            # exactly cu_seqlens_q // ratio, with no residual groups to carry.
            cu_seqlens_cmp_kv = cu_seqlens_q // ratio
            cmp_residual_kv = torch.zeros_like(cu_seqlens_cmp_kv[1:]) if ratio > 1 else None
            cmp_k_tnd = cmp_k.flatten(0, 1)
            # TND kernels consume document-local indices; cu_seqlens_cmp_kv supplies the
            # document offsets. Preserve the reference cross-document mask.
            local = _localize_indices(topk_indices, metadata.doc_ids_BL, cu_seqlens_cmp_kv)
            # Packed documents can leave leading/interleaved invalid slots;
            # compact them without changing the selected keys or their order.
            order = (local < 0).to(torch.int32).argsort(dim=-1, stable=True)
            cmp_sparse_indices = local.gather(-1, order).flatten(0, 1).to(torch.int32).unsqueeze(1).contiguous()
            cmp_k_tnd = cmp_k_tnd.unsqueeze(1).contiguous()
        output, lse = _SparseMLA.apply(
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
        )
        out = output.reshape_as(q)
        if not wants_teacher:
            return out, None
        # The kernel returns the LSE as [1, S, H]; the teacher reads [B, H, L].
        return out, lse.transpose(1, 2)
