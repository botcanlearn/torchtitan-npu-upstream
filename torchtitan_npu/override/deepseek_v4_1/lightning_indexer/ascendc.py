# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused LightningIndexer selection and its SLIKG backward edge.

The forward runs one of the two quantized ``ds41`` indexer kernels, chosen by the layer's
role in the candidate hierarchy:

- the pool source runs ``quant_lightning_indexer`` with a capacity, which returns its own
  Top-K *and* the shared candidate pool;
- a pool searcher runs ``quant_sparse_lightning_indexer``, which scores only the pool it is
  handed and returns its own Top-K;
- a layer outside the hierarchy runs ``quant_lightning_indexer`` with the pool disabled.

All three share one backward: SLIKG consumes the selection and the teacher, so the kernel
choice is a forward-only fact.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch_npu

from torchtitan_npu.models.deepseek_v4_1.indexer import FULL, Selector

_LAYOUT = "TND"
_MASK_MODE = 3  # right-down causal: entry j is complete for query t iff j < (t+1)//ratio
# The packed-MXFP4 ABI is the only value either kernel accepts; the number labels the
# storage format (two E2M1 values per byte plus uE8M0 scales), it is not a precision knob.
_QUANT_MODE = 1
# One E8M0 scale byte per 32 elements along the head dimension, on both operands.
_MX_BLOCK_SIZE = 32


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
    """The geometry the non-quantized metadata ops are built with.

    ``max_seqlen_q``/``max_seqlen_k`` are pinned to ``None`` because the legacy
    ``lightning_indexer_metadata`` takes them as optional ints; SLIKG's metadata op has the
    same pair and the same reading.  Passing them makes the call independent of whatever
    default the operator ships with.
    """
    return dict(
        num_heads_q=num_heads,
        num_heads_k=1,
        head_dim=head_dim,
        topk=topk,
        max_seqlen_q=None,
        max_seqlen_k=None,
    )


def _quantized_geometry(num_heads, head_dim, topk):
    """The geometry the ``ds41`` metadata ops are built with.

    The maximum lengths are deliberately absent rather than ``None``: these operators take
    plain ints, whose default of ``-1`` already means "any length", and ``None`` for an
    ``int`` is a type error.
    """
    return dict(
        num_heads_q=num_heads,
        num_heads_k=1,
        head_dim=head_dim,
        topk=topk,
    )


def _pack_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one ``[T, H, D]`` indexer operand into the kernels' MXFP4 storage.

    Returns the ``(data, descale)`` pair the operators take.  ``float4_e2m1fn_x2`` is one
    byte holding two E2M1 values, so the data view keeps its shape and only gains a uint8
    reinterpretation; the E8M0 scales come back flat at ``D/32`` per head and the schema
    wants them folded into ``(D/64, 2)``.  The reshape is written out rather than inferred
    so a change in the vendor quantizer's return convention fails here rather than at the
    operator's shape check.
    """

    def mx_quant() -> tuple[torch.Tensor, torch.Tensor]:
        """The vendor MX quantizer, named so the packing below reads as layout work."""
        return torch_npu.npu_dynamic_mx_quant(
            x.contiguous(),
            axis=-1,
            dst_type=torch_npu.float4_e2m1fn_x2,
            block_size=_MX_BLOCK_SIZE,
            round_mode="rint",
            scale_alg=2,
            dst_type_max=0.0,
        )

    rows, heads, head_dim = x.shape
    data, scale = mx_quant()
    return (
        data.view(torch.uint8).reshape(rows, heads, head_dim // 2).contiguous(),
        scale.view(torch.uint8).reshape(rows, heads, head_dim // 64, 2).contiguous(),
    )


def _candidate_length(candidate_block_indices: torch.Tensor) -> torch.Tensor:
    """The candidate table's valid prefix length, read off the block table itself.

    ``quant_lightning_indexer`` writes ``-1`` into every slot past the row's reachable
    block count, as a contiguous tail, so counting the non-negative entries recovers the
    length the consumer's metadata wants.  The result is shaped and typed to that contract
    directly: ``(T1, 1)`` int32 on the table's own device.

    This leans on the producer's fill convention, which is worth naming because the
    consumer does not share it: ``quant_sparse_lightning_indexer`` treats
    ``candidate_block_length`` as authoritative and *ignores* the tail, and the vendor's
    own test for it writes junk past the length to prove the tail is never read.  So the
    inference is sound only while the table comes from ``quant_lightning_indexer`` -- which
    is the only producer in this chain -- and it is deliberately re-derived here rather
    than threaded so that the pool stays one cross-layer tensor.
    """
    return (candidate_block_indices >= 0).sum(dim=-1).to(torch.int32).contiguous()


class _LightningIndexerTND(torch.autograd.Function):
    """The quantized ``ds41`` indexer forward, fused with the SLIKG backward.

    The kernels' contract (A5-verified): ``q`` and ``k`` are single-key-head bf16 with
    ``index_head_dim`` 128 and leave here as caller-packed MXFP4, ``w`` is float32
    ``[T, Hi]``, the indices are document-local with ``-1`` padding, and the SLIKG metadata
    requires ``topk`` to be 512 or an integer multiple of 1024.  The forward returns the
    document-local indices, which the fused attention reads in that same coordinate system.

    The *saved* operands are the bf16 tensors, not the packed ones: SLIKG differentiates
    them, and MXFP4 is not a differentiable storage.  The quantization is therefore a
    straight-through detail of the forward -- the backward sees the values the projections
    produced, which is the usual QAT arrangement and keeps the distillation gradient
    attached to ``wq_b``, ``weights_proj`` and ``wk``.
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
        candidate_topk_blocks,
        candidate_block_size,
        is_source,
        legacy,
        candidate,
    ):
        # These arrive with the batch axis already squeezed off by the caller: an
        # autograd.Function's backward must return one gradient per input *in the shape it
        # was handed*, so the kernels' ``[T, H, D]`` layout has to be established outside
        # this node rather than inside it.
        options = _kernel_options(attention_masks, ratio)
        # Which kernel runs, and what it is handed, is one three-way role.  The invariant
        # that covers all three is hoisted here rather than restated per branch: a candidate
        # table arrives exactly when this layer is a searcher.  The source builds it and so
        # may not be handed one, a layer outside the hierarchy has none to pass on, and the
        # legacy operator has no candidate table at all.
        expects_candidate = not legacy and candidate_topk_blocks > 0 and not is_source
        if (candidate is not None) != expects_candidate:
            raise ValueError(
                "the candidate table in flight disagrees with this layer's role: "
                + (
                    "a searcher searches the pool its source built, but none was handed to it."
                    if expects_candidate
                    else "only a searcher is handed one, but this layer received one."
                )
            )

        candidate_out = None
        if legacy:
            # The pre-quantization selection: bf16 operands straight into the non-quantized
            # operator, which has no pool to build or search.  It is kept as an escape hatch
            # -- for a geometry the quantized pair refuses, or to isolate a quantization
            # change from a wiring change -- not as a supported configuration.
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
        else:
            geometry = _quantized_geometry(idx_q.shape[1], idx_q.shape[2], topk)
            idx_q_mxfp4, idx_q_scale = _pack_mxfp4(idx_q)
            idx_k_mxfp4, idx_k_scale = _pack_mxfp4(idx_k)
            if not expects_candidate:
                # The producer and a layer outside the hierarchy run the same operator; what
                # separates them is only whether this layer publishes what it built.  The
                # condition is the negation of the guard above -- a searcher is the one role
                # that consumes -- so the two cannot disagree about which layers search.
                li_metadata = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
                    **options,
                    **geometry,
                    candidate_topk_blocks=candidate_topk_blocks,
                    candidate_block_size=candidate_block_size,
                )
                # ``sparse_values`` is empty at ``return_value=False`` and the length is
                # re-derived by the searcher from the table's ``-1`` tail, so neither slot is
                # bound to a name here.
                topk_indices, _, candidate_block_indices, _ = (
                    torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer(
                        idx_q_mxfp4,
                        idx_k_mxfp4,
                        idx_w,
                        idx_q_scale,
                        idx_k_scale,
                        topk,
                        _QUANT_MODE,
                        metadata=li_metadata,
                        return_value=False,
                        candidate_topk_blocks=candidate_topk_blocks,
                        candidate_block_size=candidate_block_size,
                        **options,
                    )
                )
                # Guarded, not unconditional: with the pool off the operator still returns
                # the candidate outputs, but as 1-D empties rather than a table.  Publishing
                # one would hand the next layer something that is not a pool and trip the
                # check above, so only the producer assigns.
                #
                # The unsqueeze is the kernel ABI's ``[T, 1, capacity]`` becoming the
                # model's cross-layer ``[B, L, 1, capacity]``, and it belongs *here* rather
                # than at the caller: this is the same edge that squeezes on the way in, so
                # both translations of the table sit next to the kernels that force them.
                # A caller-side unsqueeze cannot tell a freshly built table from one being
                # passed through, so it would grow the rank once per searcher layer.
                if is_source:
                    candidate_out = candidate_block_indices.unsqueeze(0)
            else:
                # The candidate table arrives as the model's cross-layer tensor,
                # ``[B, L, 1, capacity]``, and the kernel wants the row's own
                # ``[T, 1, capacity]``.
                block_indices = candidate.squeeze(0)
                block_length = _candidate_length(block_indices)
                slig_metadata = torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer_metadata(
                    block_length,
                    **options,
                    **geometry,
                    quant_mode=_QUANT_MODE,
                    candidate_block_size=candidate_block_size,
                )
                topk_indices, _ = torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer(
                    idx_q_mxfp4,
                    idx_k_mxfp4,
                    idx_w,
                    idx_q_scale,
                    block_indices,
                    block_length,
                    topk,
                    _QUANT_MODE,
                    candidate_block_size,
                    descale_k=idx_k_scale,
                    metadata=slig_metadata,
                    return_value=False,
                    **options,
                )
                candidate_out = candidate

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
        return topk_indices, topk_scores, candidate_out

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx,
        grad_indices,
        attn_softmax_l1_norm,
        grad_candidate,
    ):
        # The indices are integer selections: never tracked, always None.  The
        # scores' gradient is the teacher the SLIKG kernel expects, because SLIKG
        # applies ``dI = Z * Y - p`` itself.  SMLAG's backward is what puts the raw
        # ``p`` on this edge; nothing else may, and a signed value here would be read
        # as the teacher and negate the indexer's gradient.
        #
        # The candidate table's gradient is ``None`` by construction too: it is an int32
        # block table, so no graph edge ever forms on it.  Both discards are named rather
        # than elided because the two used edges are told apart by name here, and because a
        # bare ``_`` cannot be repeated in a signature.
        del grad_indices, grad_candidate
        if attn_softmax_l1_norm is None:
            # No teacher reached this edge, so no operand has a gradient either.  The count
            # is the same one the labelled tail below enumerates: one entry per forward
            # input, in the same order.
            return (None,) * 11
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
        #
        # One gradient per forward input, in order: (idx_q, idx_k, idx_w, topk, ratio,
        # attention_masks, candidate_topk_blocks, candidate_block_size, is_source, legacy,
        # candidate).  The count is spelled out rather than padded because a padded count
        # drifts silently in one direction only -- PyTorch ignores extra trailing entries
        # and raises only when the list is short -- so the mismatch would surface as a
        # confusing "expected N, got M" from the engine at some later refactor instead of
        # here.
        return (
            dq,
            dk,
            dw,
            None,  # topk
            None,  # ratio
            None,  # attention_masks
            None,  # candidate_topk_blocks
            None,  # candidate_block_size
            None,  # is_source
            None,  # legacy
            None,  # candidate
        )


class AscSelector(Selector):
    """The fused QLI/QSLI selector for the V4.1 indexer layers.

    Which kernel runs is the layer's role in the candidate hierarchy, and the role is read
    off the same config the reference selector reads it from: a Full Mode layer that carries
    a capacity is the source, any other capacity-carrying layer is a searcher, and a layer
    with no capacity sits outside the hierarchy.  Full Mode is the discriminator because it
    is the role that owns the compressed KV the pool indexes, so it is the only one that can
    build a pool; a Reindex Mode layer carries a capacity but never owns the KV.

    It emits the kernel's own document-local ``topk_indices``.  The fused attention
    consumes that same coordinate system, so neither side translates: the selection
    crosses the model boundary exactly as the kernels selected it, and only ``-1`` marks an
    unused slot.

    The pool travels as the model's cross-layer tensor, ``[B, L, 1, capacity]``, on every
    path: :class:`_LightningIndexerTND` owns both translations, unsqueezing the kernel's own
    ``[T, 1, capacity]`` where it produces one and squeezing it back where a searcher hands
    it to QSLI.  Keeping them on one edge is what stops the rank from growing per searcher
    layer, since a pass-through table is indistinguishable from a fresh one at the caller.
    Its length is not carried -- a searcher recovers it from the table's ``-1`` tail (see
    :func:`_candidate_length`).

    ``consumes_indexer_teacher`` is what pairs this node with the port that emits the
    teacher (``sparse_attn.asc``); see
    :func:`~torchtitan_npu.models.deepseek_v4_1.attention._check_indexer_teacher_pair`.

    The ``legacy`` switch swaps the whole selection back to the pre-quantization operator.
    It is a deployment choice rather than a model fact -- see :class:`AscSelector.Config`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Selector.Config):
        consumes_indexer_teacher: ClassVar[bool] = True
        # Select the pre-quantization, pool-free ``lightning_indexer`` instead of the
        # quantized ``ds41`` pair.  The field's own default is the pool-free path, and the
        # override entry's is the same, so a run reaches the quantized pair only by asking:
        #
        #   --override.imports \
        #     'torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc={"legacy":false}'
        #
        # The pool-free path has no candidate pool, so a run that takes it scores every
        # visible entry on every indexer layer -- which is what the fused path did before the
        # quantized pair landed, and is the baseline the pooled path is measured against.
        legacy: bool = True

    def __init__(self, config: Config):
        super().__init__(config)
        self.legacy = config.legacy

    def forward(
        self,
        idx_q_BLHiDi: torch.Tensor,
        idx_k_BNDi: torch.Tensor,
        weights_BLHi: torch.Tensor,
        attention_masks,
        *,
        candidates_BL1C: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        topk = self.index_topk
        # Only a Full Mode layer owns the compressed KV the pool indexes, so it is the only
        # role that can build one.  That is the same pair -- Full Mode plus a capacity --
        # the reference selector dispatches on, so both paths read one rule off one config.
        is_source = self.mode is FULL and self.has_candidate_pool
        topk_indices, topk_scores, candidate = _LightningIndexerTND.apply(
            # The batch axis is the packed row, so it is squeezed rather than indexed: every
            # kernel in this path addresses one row.
            idx_q_BLHiDi.squeeze(0),
            idx_k_BNDi.squeeze(0).unsqueeze(1),
            weights_BLHi.squeeze(0).float(),
            topk,
            self.compress_ratio,
            attention_masks,
            self.candidate_topk_blocks,
            self.candidate_block_size,
            is_source,
            self.legacy,
            candidates_BL1C,
        )
        # Two layout translations and nothing else: ``_LightningIndexerTND`` already
        # emitted the selection sorted, in the kernels' ``[T, 1, K]`` layout, and the model
        # contract is ``[1, L, K]``.  Sorting at this level instead would desynchronise the
        # saved indices from the teacher (see the Function).
        #
        # The candidate table needs no translation here: it is already the model's
        # cross-layer ``[B, L, 1, capacity]`` on both paths, because the Function converts
        # the kernel's own ``[T, 1, capacity]`` on the edge that produced it.
        topk_indices_BLK = topk_indices.reshape(1, -1, topk)
        topk_scores_BLK = topk_scores.reshape(1, -1, topk)
        candidates_BL1C = candidate
        return topk_indices_BLK, topk_scores_BLK, candidates_BL1C
