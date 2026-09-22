# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused LightningIndexer selection and its SLIKG backward edge."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from torchtitan_npu.models.deepseek_v4_1.indexer import Selector

_LAYOUT = "TND"
_MASK_MODE = 3  # right-down causal: entry j is complete for query t iff j < (t+1)//ratio


def _kernel_options(attention_masks, ratio):
    """The kernel options, read off the metadata's frames.

    Both the metadata call and the main call take these, and they must agree
    option-for-option; reading one source is what makes that structural rather than a
    convention two call sites have to keep in step.
    """
    compressed = attention_masks.kernel.frame_for(ratio)
    return dict(
        cu_seqlens_q=attention_masks.kernel.q.cu_seqlens,
        cu_seqlens_k=compressed.cu_seqlens,
        cmp_residual_k=compressed.residual,
        layout_q=_LAYOUT,
        layout_k=_LAYOUT,
        mask_mode=_MASK_MODE,
        cmp_ratio=ratio,
    )


def _kernel_geometry(num_heads, head_dim, topk):
    """The geometry both metadata calls are built with."""
    return dict(
        num_heads_q=num_heads,
        num_heads_k=1,
        head_dim=head_dim,
        topk=topk,
        max_seqlen_q=None,
        max_seqlen_k=None,
    )


class _LightningIndexerTND(torch.autograd.Function):
    """LI v2 forward fused with the SLIKG backward, in the TND varlen layout.

    The kernels' contract (A3-verified): ``q``/``k`` are BF16 with a single
    key head and ``index_head_dim`` 128, ``w`` is float32 ``[T, Hi]``, the
    indices are document-local with ``-1`` padding, and the SLIKG metadata
    requires ``topk`` to be 512 or an integer multiple of 1024.  The forward
    returns the document-local indices, which the fused attention reads in that
    same coordinate system.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        idx_q,
        idx_k,
        idx_w,
        topk,
        ratio,
        attention_masks,
    ):
        options = _kernel_options(attention_masks, ratio)
        li_metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
            **_kernel_geometry(idx_q.shape[1], idx_q.shape[2], topk),
            **options,
        )
        topk_indices, _ = torch.ops.cann_ops_transformer.lightning_indexer(
            idx_q,
            idx_k,
            idx_w,
            topk,
            metadata=li_metadata,
            return_value=1,
            **options,
        )
        # The kernel's own order is unspecified, so the selection is sorted into position
        # order here -- inside the Function, before anything is saved.  That placement is
        # the whole point: the teacher arrives on ``topk_scores``' *gradient*, positioned
        # per slot by whatever selection SMLA was handed, and SLIKG consumes it next to
        # ``topk_indices``.  Sorting after ``apply`` returns would leave the saved indices
        # in kernel order while the teacher is in sorted order, and SLIKG would pair each
        # slot's mass with a different key.  Sorting here keeps the two in step by
        # construction, and the kernel layout stays in one place.
        #
        # Descending gives both properties with one sort: the valid entries come out in
        # descending position order and ``-1``, being the smallest index, lands behind
        # every one of them.  (The reference sorts ascending instead; neither the kernel
        # schema nor the operator docs fix a direction.)
        topk_indices = topk_indices.sort(dim=-1, descending=True).values
        # The carrier for the teacher edge, fabricated rather than taken from the kernel's
        # second output.  The gradient this Function's backward receives on it *is* the
        # teacher -- ``attn_softmax_l1_norm`` there feeds SLIKG -- so it has to be an output
        # of this Function that the engine differentiates through, and returning it from
        # here is what supplies that ``grad_fn``.  A tensor built by the caller after
        # ``apply`` returns cannot have it and leaves the indexer untrained.
        #
        # Nothing reads its value: SMLAG replaces it wholesale and the attention's core
        # accepts and ignores it.  Shape, dtype and the edge are the whole contract, which
        # is why the kernel's own scores are dropped rather than permuted across to here.
        # The shape is the selection's own, ``[T, 1, K]`` -- SLIKG requires the teacher to
        # match ``sparse_indices`` exactly.
        topk_scores = topk_indices.new_empty(topk_indices.shape, dtype=torch.float32, requires_grad=True)
        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices)
        ctx.topk, ctx.ratio = topk, ratio
        # The mask is a dataclass, so ``save_for_backward`` cannot take it; it is a
        # context attribute instead.  That is not extra retention: it holds the same
        # boundary tensors the graph already keeps for the backward, and the node -- and
        # with it this reference -- is released when that backward has run.
        ctx.attention_masks = attention_masks
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, _, attn_softmax_l1_norm):  # pyrefly: ignore [bad-override]
        # The indices are integer selections: never tracked, always None.  The
        # scores' gradient is the teacher the SLIKG kernel expects, because SLIKG
        # applies ``dI = Z * Y - p`` itself.  SMLAG's backward is what puts the raw
        # ``p`` on this edge; nothing else may, and a signed value here would be read
        # as the teacher and negate the indexer's gradient.
        if attn_softmax_l1_norm is None:
            return None, None, None, None, None, None, None
        # Nothing but the operands is saved.  The options and the geometry are read back
        # off ``ctx.attention_masks`` and the saved tensors -- the same source the forward
        # used, which is what keeps the two kernel calls from drifting apart -- and the
        # SLIKG metadata is built from them here.
        idx_q, idx_k, idx_w, topk_indices = ctx.saved_tensors
        # The KL objective is a mean over query tokens, and the trainer's
        # ``global_valid_tokens`` division cannot reach it: the teacher never passes
        # through the loss function, it is injected straight into SLIKG on this edge.  So
        # the token normalisation is applied here, to the only quantity that trains the
        # indexer.
        #
        # ``idx_q`` is the TND query and its leading axis is the packed token count, so
        # this is the row's ``seqlen``.  Scaling ``p`` is exactly scaling ``dI``: SLIKG's
        # ``dI = Z * Y - p`` is affine in ``p``, so ``dI(p/S) = dI(p)/S`` and the kernel's
        # own ``dq``/``dk``/``dw`` come out normalised with no rescaling afterwards.
        attn_softmax_l1_norm = attn_softmax_l1_norm / idx_q.shape[0]
        options = _kernel_options(ctx.attention_masks, ctx.ratio)
        slig_metadata = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
            **_kernel_geometry(idx_q.shape[1], idx_q.shape[2], ctx.topk),
            **options,
        )
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            metadata=slig_metadata,
            **options,
        )
        # ``idx_w`` arrives float32 (the kernel contract); the cast back to the
        # projection dtype happens in the eager graph outside the Function.
        return dq, dk, dw, None, None, None, None


class AscSelector(Selector):
    """The fused LI/SLIKG selector for the V4.1 indexer layers.

    It emits the kernel's own document-local ``topk_indices``.  The fused attention
    consumes that same coordinate system, so neither side translates: the selection
    crosses the model boundary exactly as SLIKG selected it, and only ``-1`` marks an
    unused slot.

    ``consumes_indexer_teacher`` is what pairs this node with the port that emits the
    teacher (``sparse_attn.asc``); see
    :func:`~torchtitan_npu.models.deepseek_v4_1.attention._check_indexer_teacher_pair`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Selector.Config):
        consumes_indexer_teacher: ClassVar[bool] = True

    def forward(
        self,
        idx_q_BLHiDi: torch.Tensor,
        idx_k_BNDi: torch.Tensor,
        weights_BLHi: torch.Tensor,
        attention_masks,
        *,
        candidates_BLN: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        ratio = self.compress_ratio
        topk = self.index_topk
        topk_indices, topk_scores = _LightningIndexerTND.apply(
            idx_q_BLHiDi.squeeze(0),
            idx_k_BNDi.squeeze(0).unsqueeze(1),
            weights_BLHi.squeeze(0).float(),
            topk,
            ratio,
            attention_masks,
        )
        # Only a layout translation here: ``_LightningIndexerTND`` already emitted both
        # tensors sorted, in the kernels' ``[T, 1, K]`` layout, and the model contract is
        # ``[1, L, K]``.  Sorting at this level instead would desynchronise the saved
        # indices from the teacher (see the Function).
        topk_indices_BLK = topk_indices.reshape(1, -1, topk)
        topk_scores_BLK = topk_scores.reshape(1, -1, topk)
        return topk_indices_BLK, topk_scores_BLK, candidates_BLN
