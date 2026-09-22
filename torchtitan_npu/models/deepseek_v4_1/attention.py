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

Packed documents are isolated by ``doc_ids``, the metadata's only per-token field: the
operator applies it to the sliding-window branch, and the indexer uses it to keep its
top-``K`` inside the query's own document (the operator's contract leaves the sparse
branch to the caller).  ``doc_ids`` is not an attention input: the indexer derives its
entry-axis isolation from it while building the precomputed selection masks.  ``positions``
is a per-forward argument, not metadata, and only drives RoPE.

The indexer's teacher belongs to a port, not to this core.  It is the operator's per-head
log-sum-exp of the full softmax (window, selected compressed entries and sink), turned
into the raw marginal mass ``p`` and emitted on ``topk_scores``' gradient, which the
indexer's SLIKG backward consumes to train the indexer.  A query row with no reachable
compressed entry contributes nothing.  Nothing in this file holds a loss.

The projections use the rope module with the site's un-rotated prefix width.
The reference forward owns Attention Gym attention; fused ports replace the complete
inner-attention forward.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from attn_gym.sparse.selected_attention import selected_attention
from torch import nn
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .compressor import Compressor
from .indexer import HierarchicalIndexer


def _check_indexer_teacher_pair(
    attn_cfg: "CompressedSparseInnerAttention2.Config", indexer_cfg: HierarchicalIndexer.Config
) -> None:
    """The fused indexer teacher is a pair: one port emits it, one consumes it.

    ``topk_scores`` is carried only between the two kernel backwards.  The emitting
    port writes the raw teacher on ``topk_scores``' gradient; the consuming selector
    is the only thing that ever reads that edge.  Fusing one side alone therefore
    breaks the objective -- silently, since neither half raises on its own -- so the
    mismatch is rejected here instead of being discovered from a flat training curve.

    Neither half is a usable fallback, and this check is the only thing that keeps
    them from being reached:

    - Selector without the fused core: ``topk_scores``' gradient is ``None``, so SLIKG
      returns zero gradient and the indexer never updates.
    - Fused core without the fused selector: the reference selector does pick the
      teacher up off that edge, but it applies attention-level marginals to summed
      per-head indexer logits.  The two sides are not the same quantity, so the
      resulting training signal is wrong rather than merely different.  That path is
      deliberately left unfixed -- it is unreachable while this check is in place.

    Both halves are compile-time facts on the config classes, so this runs on the
    configs rather than on the built modules: ``Module`` does not retain its config.
    A ratio-0 layer is skipped: it has no second KV stream, so neither half has an
    edge to offer or consume.
    """
    if attn_cfg.compress_ratio == 0:
        return
    fused_core = getattr(type(attn_cfg), "provides_indexer_teacher", False)
    fused_selector = getattr(type(indexer_cfg.selector), "consumes_indexer_teacher", False)
    if fused_core == fused_selector:
        return
    if fused_core:
        active, missing = "sparse_attn.asc", "lightning_indexer.asc"
        consequence = "the reference selector would take the teacher onto summed per-head logits"
    else:
        active, missing = "lightning_indexer.asc", "sparse_attn.asc"
        consequence = "topk_scores would carry no teacher"
    raise ValueError(
        f"A fused V4.1 attention core and a fused selector must be enabled together: {active} is "
        f"active without {missing}, so {consequence} and the indexer cannot be trained correctly. "
        "Enable both or neither. See torchtitan_npu/override/README.md#deepseek-v41."
    )


class CompressedSparseInnerAttention2(Module):
    """CSA2's sparse core: sliding window plus the selected compressed entries.

    The whole core is Attention Gym's ``selected_attention`` in one call: ``q`` carries
    ``H`` heads while both KV sources carry a single shared head, and ``topk_indices``
    index the compressed KV with ``-1`` for unused slots.  ``compress_ratio`` carries
    the layer's verified ratio (0 window-only, 1 a full-resolution second KV stream,
    2 the compressed shared KV): a fused port needs it to translate the metadata's
    document boundaries into its kernel layout, so it is never guessed from shapes.

    A port that can produce the indexer's teacher replaces this forward wholesale and
    emits it on ``topk_scores``' gradient; this core has no kernel to make one.  Such a
    port's :class:`Config` sets ``provides_indexer_teacher``, which is what
    :func:`_check_indexer_teacher_pair` pairs against the selector's
    ``consumes_indexer_teacher``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # Declared on the config, not the module: the pair check runs on the config
        # tree before anything is built, and ``Module`` does not keep its config.
        provides_indexer_teacher: ClassVar[bool] = False

        window_size: int
        softmax_scale: float
        # The layer's compression ratio, carried explicitly for the fused ports.
        compress_ratio: int

    def __init__(self, config: Config):
        super().__init__()
        self.window_size = config.window_size
        self.softmax_scale = config.softmax_scale
        self.compress_ratio = config.compress_ratio

    def forward(
        self,
        q: torch.Tensor,
        swa_k: torch.Tensor,
        attn_sink: torch.Tensor,
        attention_masks,
        *,
        cmp_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            q: Queries of shape ``[B, L, H, Dk]``.
            swa_k: Sliding-window KV of shape ``[B, L, Dk]``, shared across heads.
            attn_sink: Per-head sink logits of shape ``[H]``.
            attention_masks: The forward's varlen metadata; the operator reads its
                document ids for the sliding-window branch.
            cmp_k: Shared compressed KV of shape ``[B, N, Dk]``.
            topk_indices: Selected compressed entries ``[B, L, K]``, ``-1`` for unused.
            topk_scores: Accepted for the forward contract and unused here; it exists so
                a port that produces the indexer's teacher has somewhere to carry it.

        Returns:
            Attention output of shape ``[B, L, H, Dk]``.

        This core is the plain reference: it computes attention and nothing else.  The
        indexer's distillation objective is not here -- it needs this operator's softmax
        denominator, and a port that can produce it emits the teacher on ``topk_scores``'
        gradient instead of returning it.
        """
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together.")

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
            doc_ids=attention_masks.ref.doc_ids_BL,
            sliding_window_size=self.window_size,
            scale=self.softmax_scale,
            # TODO: run the fused kernels once they validate this path; ``impl`` is the
            # pinned attn-gym 0.0.9 argument, ``"reference"`` its eager PyTorch path.
            impl="reference",
        )
        assert isinstance(result, torch.Tensor)
        return result.transpose(1, 2)


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
        _check_indexer_teacher_pair(cfg.inner_attention, cfg.indexer)

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
            self.attn_sink,
            attention_masks,
            cmp_k=cmp_k if uses_cmp else None,
            topk_indices=topk_indices if uses_cmp else None,
            topk_scores=topk_scores if uses_cmp else None,
        )
        o = self.rope(o, positions=positions, inverse=True)

        # The output projection is grouped: wo_a projects each query-head group on its
        # own, wo_b mixes the per-group results back to the model dimension.  The group
        # count comes from the module so a group-wise sharding can narrow it later.
        o = self.wo_a(o.reshape(bsz * seqlen, self.wo_a.n_batches, -1))
        return (self.wo_b(o.flatten(-2)), cmp_k, idx_k, topk_indices, topk_scores, candidates)
