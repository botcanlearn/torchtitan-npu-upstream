# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CSA2 attention for DeepSeek V4.1.

Shape legend for this file:
    B = batch, L = sequence length, D = model dimension,
    H = ``n_heads``, Dk = ``head_dim``, rd = ``rope_head_dim``,
    N = number of compressed KV entries, K = ``index_topk``.

Each query attends to two sources at once, combined into a single masked softmax by
Attention Gym's ``selected_attention``: its own sliding window over the layer's KV
``[B, L, Dk]``, and the ``K`` compressed entries selected by the indexer out of the
shared compressed KV ``[B, N, Dk]``.  A learned per-head sink logit takes part in the
softmax denominator without contributing a value, so a row with no reachable entry
still produces zeros instead of NaN.

Packed documents are isolated by the metadata's ``doc_ids``, which is the only varlen
metadata: the operator applies it to the sliding-window branch, and the indexer uses it
to keep its top-``K`` inside the query's own document (the operator's contract leaves
the sparse branch to the caller).  ``positions`` still drives RoPE; it is not segment
metadata.

The attention is also where the indexer's distillation loss is applied.  The operator
returns the per-head log-sum-exp of the full softmax (window, selected compressed entries
and sink) and the student logits arrive as an input; the loss itself, including the
teacher it rebuilds from them, lives on ``IndexerDistillLoss``.  The loss's inputs are
detached at that call site: the distillation must train the indexer and nothing else.
A query row the metadata marks as structural padding is excluded from the
distillation instead of contributing.

The projections use the rope module with the site's un-rotated prefix width.
The reference forward owns Attention Gym attention and teacher reconstruction;
fused ports replace the complete inner-attention forward.
"""

from dataclasses import dataclass

import torch
from attn_gym.sparse.selected_attention import AuxRequest, selected_attention
from torch import nn
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.aux_loss import LoggedAuxLoss
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor
from .indexer import HierarchicalIndexer


class CompressedSparseInnerAttention2(Module):
    """CSA2's sparse core: sliding window plus the selected compressed entries.

    The whole core is Attention Gym's ``selected_attention`` in one call: ``q`` carries
    ``H`` heads while both KV sources carry a single shared head, and ``topk_indices``
    index the compressed KV with ``-1`` for unused slots.  ``compress_ratio`` carries
    the layer's verified ratio (0 window-only, 1 a full-resolution second KV stream,
    2 the compressed shared KV): a fused port needs it to translate the metadata's
    document boundaries into its kernel layout, so it is never guessed from shapes.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        softmax_scale: float
        # The layer's compression ratio, carried explicitly for the fused ports.
        compress_ratio: int
        # Indexer distillation loss, attached only on layers that consume the selection
        # (``compress_ratio > 0`` with an index source at or before them).
        aux_loss: LoggedAuxLoss.Config | None = None
        score_gradient: str = "logits"

    def __init__(self, config: Config):
        super().__init__()
        self.window_size = config.window_size
        self.softmax_scale = config.softmax_scale
        self.compress_ratio = config.compress_ratio
        if config.score_gradient not in ("logits", "teacher"):
            raise ValueError(f"Unknown scores gradient mode: {config.score_gradient}")
        self.score_gradient = config.score_gradient
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def forward(
        self,
        q: torch.Tensor,
        swa_k: torch.Tensor,
        cmp_k: torch.Tensor | None = None,
        *,
        attention_masks,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        attn_sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            q: Queries of shape ``[B, L, H, Dk]``.
            swa_k: Sliding-window KV of shape ``[B, L, Dk]``, shared across heads.
            cmp_k: Shared compressed KV of shape ``[B, N, Dk]``.
            attention_masks: The forward's varlen metadata; the operator reads its
                document ids for the sliding-window branch and the loss reads its
                valid-token marks when the loader produced any.
            topk_indices: Selected compressed entries ``[B, L, K]``, ``-1`` for unused.
            topk_scores: Student logits at those entries ``[B, L, K]``.
            attn_sink: Per-head sink logits of shape ``[H]``.

        Returns:
            Attention output of shape ``[B, L, H, Dk]``.
        """
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together.")

        # The loss runs only while training: in eval there is nothing to distill and the
        # extra ``lse`` the teacher needs would be dead output.  A compress layer that
        # consumes the selection always has the teacher's other inputs -- ``cmp_k`` holds
        # the scored entries and an index source precedes it -- which is exactly when the
        # loss is attached.
        aux_loss = self.aux_loss
        wants_teacher = self.training and aux_loss is not None and cmp_k is not None
        batch, num_tokens, _, head_dim = q.size()
        local_kv_B1LD = swa_k.reshape(batch, 1, num_tokens, head_dim)
        if cmp_k is not None:
            # ``forward`` pairs the two inputs, so ``topk_indices`` is present here.
            assert topk_indices is not None
            sparse_kv_B1ND = cmp_k.reshape(batch, 1, cmp_k.size(1), head_dim)
            kv_indices_BLK = topk_indices
        else:
            # Window-only layer: the sparse pool is empty and every query keeps the
            # window plus the sink.
            sparse_kv_B1ND = q.new_zeros(batch, 1, 0, head_dim)
            kv_indices_BLK = torch.empty(batch, num_tokens, 0, dtype=torch.long, device=q.device)

        result = selected_attention(
            q.transpose(1, 2),
            local_kv_B1LD,
            sparse_kv_B1ND,
            kv_indices_BLK,
            attention_sink=attn_sink,
            doc_ids=attention_masks.doc_ids_BL,
            sliding_window_size=self.window_size,
            scale=self.softmax_scale,
            # TODO: run the fused kernels once they validate this path; ``impl`` is the
            # pinned attn-gym 0.0.9 argument, ``"reference"`` its eager PyTorch path.
            impl="reference",
            return_aux=AuxRequest(lse=True) if wants_teacher else None,
        )
        if not wants_teacher:
            assert isinstance(result, torch.Tensor)
            return result.transpose(1, 2)
        assert isinstance(result, tuple)
        attn_BHLD, aux = result
        attn_BLHD, lse_BHL = attn_BHLD.transpose(1, 2), aux.lse
        # ``wants_teacher`` paired cmp_k with topk_indices, so all three are present.
        assert aux_loss is not None
        assert lse_BHL is not None
        # The teacher's sources are constants: the distillation must train the indexer and
        # nothing else, and ``topk_scores`` is the one live input, carrying the gradient
        # that trains it.  ``lse`` arrives as ``[B, H, L]``; the loss reads ``[B, L, H]``.
        return aux_loss(
            q.detach(),
            cmp_k.detach(),
            topk_indices,
            lse_BHL.transpose(1, 2).detach(),
            topk_scores,
            carrier=attn_BLHD,
            query_valid_mask=attention_masks.valid_tokens_BL,
        )


class Attention(BaseAttention):
    """Latent attention with a grouped output projection, CSA2's per-layer wrapper.

    Projections, both RoPE phases, the compressor and the indexer live here; the sparse
    core is the ``inner_attention`` module.  The shared cross-layer tensors (compressed
    KV, index keys, selected entries, student logits, candidate pool) are threaded
    through as ordinary inputs and returned updated, so a layer's role is visible at the
    call site rather than hidden in mutable state.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        head_dim: int
        # The pinned V3 checkpoint mapping branches on ``q_lora_rank`` (it selects the
        # LoRA-style HF keys), so the field stays even though this module only needs the
        # projection configs.
        q_lora_rank: int
        compress_ratio: int
        inner_attention: CompressedSparseInnerAttention2.Config  # pyrefly: ignore [bad-override]
        rope: RoPE.Config
        compressor: Compressor.Config
        indexer: HierarchicalIndexer.Config
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: BatchedLinear.Config
        wo_b: Linear.Config

    def __init__(self, config: "Attention.Config"):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.compress_ratio = cfg.compress_ratio
        self.rope = cfg.rope.build()
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        # One sink logit per head, fp32 as in the released checkpoint and the kernels.
        self.attn_sink = nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))
        self.compressor = cfg.compressor.build()
        self.indexer = cfg.indexer.build()
        self.inner_attention = cfg.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_masks,
        *,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(o, cmp_k, idx_k, topk_indices, topk_scores, candidates)``.

        The returned shared tensors are this layer's contribution to the chain: freshly
        computed where the layer is a source, otherwise the inputs unchanged.
        """
        bsz, seqlen, _ = x.size()

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(bsz, seqlen, -1, self.head_dim)
        swa_k = self.kv_norm(self.wkv(x))

        # Every layer compresses and indexes: a source publishes new tensors, a reusing
        # layer hands back the ones in flight (asserted inside those modules).
        cmp_k, latent = self.compressor(x, positions, cmp_k)
        # The indexer is trained by distillation alone, so it reads the trunk as
        # constants.  The shared index keys are the exception: a Reindex Mode layer keeps
        # the ``idx_k`` it was handed live, so its consumers go on training the key's
        # owner.
        idx_k, topk_indices, topk_scores, candidates = self.indexer(
            x.detach(),
            qr.detach(),
            positions,
            attention_masks,
            latent=latent.detach() if latent is not None else None,
            idx_k=idx_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            candidates=candidates,
        )

        # The rope config carries the un-rotated prefix width, so the module rotates the
        # trailing span and puts the prefix back.
        q = self.rope(q, positions=positions)
        # The shared KV latent is one rank-2 head; RoPE rotates rank-3 [B, L, N, H].
        swa_k = self.rope(swa_k.unsqueeze(2), positions=positions).squeeze(2)

        uses_cmp = self.compress_ratio > 0
        o = self.inner_attention(
            q,
            swa_k,
            cmp_k if uses_cmp else None,
            attention_masks=attention_masks,
            topk_indices=topk_indices if uses_cmp else None,
            topk_scores=topk_scores if uses_cmp else None,
            attn_sink=self.attn_sink,
        )
        o = self.rope(o, positions=positions, inverse=True)

        # The output projection is grouped: wo_a projects each query-head group on its
        # own, wo_b mixes the per-group results back to the model dimension.  The group
        # count comes from the module so a group-wise sharding can narrow it later.
        o = self.wo_a(o.reshape(bsz * seqlen, self.wo_a.n_batches, -1))
        return (self.wo_b(o.flatten(-2)), cmp_k, idx_k, topk_indices, topk_scores, candidates)
