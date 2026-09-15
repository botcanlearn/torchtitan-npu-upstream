# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 numerical-reference DSA (eager, per-document).

An eager per-document implementation over the packed stream, matching the
inference baseline op-for-op: FP32 gather-matmul attention with a per-head
sink, per-document window indices, and the materialized shared/compressed
KV.  This is the production attention core of V4.1 — not a debug path.

Query chunking (``TTNPU_DSA_ATTN_CHUNK``, default 256) bounds the
broadcast products along the query axis without changing any reduction
axis; the chunk size is part of the frozen backward contract (it groups
the kv/sink gradient accumulation).
"""

import itertools
import os
from dataclasses import dataclass, field

import torch
from torchtitan.models.common.attention import FlexAttention, VarlenAttention

from .metadata import CompressedVarlenMetadata
from .reference import ReferenceCompressedVarlenMetadata

_ATTN_CHUNK = int(os.environ.get("TTNPU_DSA_ATTN_CHUNK", "256"))


def _window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    device,
) -> torch.Tensor:
    window = min(seqlen, window_size)
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    idxs = (base - window + 1).clamp(0) + torch.arange(window, device=device)
    idxs = torch.where(idxs > base, -1, idxs)
    return idxs.unsqueeze(0).expand(bsz, -1, -1)


def _sparse_attn_golden_chunk(
    q_BMHD: torch.Tensor,
    kv_BND: torch.Tensor,
    attn_sink_H: torch.Tensor,
    topk_idxs_BMK: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Exact pure-Torch operation order used by the inference reference."""
    batch, seqlen, heads, head_dim = q_BMHD.shape
    safe = topk_idxs_BMK.clamp_min(0).long()
    gathered = kv_BND[:, None].expand(batch, seqlen, kv_BND.size(1), head_dim)
    gathered = gathered.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, head_dim))
    score = (q_BMHD.unsqueeze(2) * gathered.unsqueeze(3)).sum(dim=-1) * softmax_scale
    score = score.permute(0, 1, 3, 2)
    score = score.masked_fill(topk_idxs_BMK.unsqueeze(2) < 0, -torch.inf)
    sink = attn_sink_H.view(1, 1, heads, 1).expand(batch, seqlen, heads, 1)
    score = torch.cat([score, sink], dim=-1)
    prob = score.softmax(dim=-1)
    return (prob[..., :-1].unsqueeze(-1) * gathered.unsqueeze(2)).sum(dim=3).to(q_BMHD.dtype)


def _sparse_attn_golden(q_BMHD, kv_BND, attn_sink_H, topk_idxs_BMK, softmax_scale):
    """Bound broadcast products along queries without changing reduction axes."""
    m = q_BMHD.size(1)
    chunk = _ATTN_CHUNK if _ATTN_CHUNK > 0 else m
    return torch.cat(
        [
            _sparse_attn_golden_chunk(
                q_BMHD[:, start : start + chunk],
                kv_BND,
                attn_sink_H,
                topk_idxs_BMK[:, start : start + chunk],
                softmax_scale,
            )
            for start in range(0, m, chunk)
        ],
        dim=1,
    )


def _sequence_ranges(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    bounds = cu_seqlens.tolist()
    return list(itertools.pairwise(bounds))


def _localize_precomputed_indices(
    topk_indices: torch.Tensor,
    metadata: CompressedVarlenMetadata,
    ratio: int,
) -> torch.Tensor:
    """Convert container-grid indices into the current document's local grid."""
    if topk_indices.ndim == 4 and topk_indices.shape[2] == 1:
        topk_indices = topk_indices.squeeze(2)
    if topk_indices.ndim != 3:
        raise ValueError("precomputed sparse indices must have shape [B, L, K] or [B, L, 1, K]")
    cu_q = metadata.varlen.cu_seq_q.to(device=topk_indices.device, dtype=torch.long)
    lengths = torch.diff(cu_q)
    query_docs = torch.repeat_interleave(torch.arange(lengths.numel(), device=topk_indices.device), lengths)
    if ratio > 1:
        plan = metadata.plans.get(ratio)
        if plan is None or plan.cu_seqlens_cmp_k is None:
            raise ValueError(f"missing compressed plan for ratio={ratio}")
        starts = plan.cu_seqlens_cmp_k.to(device=topk_indices.device, dtype=torch.long)
    else:
        starts = cu_q
    doc_start = starts[query_docs]
    doc_end = starts[query_docs + 1]
    indices = topk_indices.reshape(-1, topk_indices.shape[-1]).to(torch.long)
    valid = (indices >= doc_start.unsqueeze(-1)) & (indices < doc_end.unsqueeze(-1))
    return torch.where(valid, indices - doc_start.unsqueeze(-1), -1).view_as(topk_indices)


class V41SparseAttention(FlexAttention):
    """The V4.1 per-document reference DSA over packed varlen metadata."""

    # The Config intentionally derives from VarlenAttention.Config (not
    # FlexAttention.Config): the decoder's get_attention_masks dispatches on
    # the inner-attention config type, and the packed varlen stream is the
    # V4.1 contract.
    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):  # pyrefly: ignore [bad-override]
        window_size: int  # pyrefly: ignore [bad-override]
        compress_ratio: int
        softmax_scale: float
        index_topk: int
        kernel_options: dict = field(default_factory=dict)

    def __init__(self, config: Config) -> None:
        super().__init__(config)  # pyrefly: ignore [bad-argument-type]
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        sparse_indices=None,
        attn_sink=None,
        *,
        attention_masks: ReferenceCompressedVarlenMetadata | None = None,
        compress_ratio: int | None = None,
    ):
        if not isinstance(attention_masks, CompressedVarlenMetadata):
            raise TypeError(
                f"V41SparseAttention requires CompressedVarlenMetadata attention masks, got {type(attention_masks)}."
            )
        metadata = attention_masks
        ratio = self.compress_ratio if compress_ratio is None else compress_ratio
        query = q.flatten(0, 1)
        original_kv = swa_k.flatten(0, 1)
        shared_full = ratio == 1 and cmp_k is not None

        precomputed_topk = sparse_indices
        if precomputed_topk is not None:
            precomputed_topk = _localize_precomputed_indices(precomputed_topk, metadata, ratio)

        plan = metadata.plans.get(ratio)
        if cmp_k is None or (not shared_full and plan is None):
            compressed_kv = query.new_empty((0, query.shape[-1]))
        elif shared_full:
            compressed_kv = cmp_k.flatten(0, 1)
        else:
            if plan.cu_seqlens_cmp_k is None:  # pyrefly: ignore [missing-attribute]
                raise ValueError(f"ratio={ratio} compressed KV requires cu_seqlens_cmp_k")
            compressed_kv = cmp_k.flatten(0, 1)[: plan.cu_seqlens_cmp_k[-1]]

        if precomputed_topk is not None:
            compressed_indices = precomputed_topk.reshape(-1, precomputed_topk.shape[-1])
        elif ratio > 1:
            raise ValueError(f"V4.1 ratio={ratio} attention requires precomputed sparse indices")
        else:
            compressed_indices = None

        outputs = []
        compressed = None if ratio <= 1 else metadata.plans.get(ratio)
        block_ranges = (
            None
            if compressed is None
            else _sequence_ranges(compressed.cu_seqlens_cmp_k)  # pyrefly: ignore [bad-argument-type]
        )
        for document_id, (q_start, q_end) in enumerate(_sequence_ranges(metadata.varlen.cu_seq_q)):
            length = q_end - q_start
            document_query = query[q_start:q_end].unsqueeze(0)
            document_kv = original_kv[q_start:q_end]
            indices = _window_topk_idxs(self.window_size, 1, length, query.device)
            if shared_full and precomputed_topk is None:
                document_compressed = compressed_kv[q_start:q_end]
                local = torch.arange(length, device=query.device)
                document_indices = local.unsqueeze(0).expand(length, -1)
                causal = document_indices <= local.unsqueeze(1)
                document_indices = torch.where(
                    causal,
                    document_indices,
                    torch.full_like(document_indices, -1),
                ).unsqueeze(0)
                indices = torch.cat([indices, document_indices], dim=-1)
                document_kv = torch.cat([document_kv, document_compressed], dim=0)
            elif shared_full:
                document_compressed = compressed_kv[q_start:q_end]
                document_indices = precomputed_topk.reshape(  # pyrefly: ignore [missing-attribute]
                    -1,
                    precomputed_topk.shape[-1],  # pyrefly: ignore [missing-attribute]
                )[  # pyrefly: ignore [missing-attribute]
                    q_start:q_end
                ]  # pyrefly: ignore [missing-attribute]
                document_indices = torch.where(  # pyrefly: ignore [no-matching-overload]
                    document_indices < 0,
                    document_indices,
                    document_indices + length,
                ).unsqueeze(0)
                indices = torch.cat([indices, document_indices], dim=-1)
                document_kv = torch.cat([document_kv, document_compressed], dim=0)
            elif compressed is not None:
                c_start, c_end = block_ranges[document_id]  # pyrefly: ignore [unsupported-operation]
                document_compressed = compressed_kv[c_start:c_end]
                document_indices = compressed_indices[  # pyrefly: ignore [unsupported-operation]
                    q_start:q_end, : c_end - c_start
                ]  # pyrefly: ignore [unsupported-operation]
                document_indices = torch.where(  # pyrefly: ignore [no-matching-overload]
                    document_indices < 0,
                    document_indices,
                    document_indices + length,
                ).unsqueeze(0)
                indices = torch.cat([indices, document_indices], dim=-1)
                document_kv = torch.cat([document_kv, document_compressed], dim=0)
            result = _sparse_attn_golden(
                document_query,
                document_kv.unsqueeze(0),
                attn_sink,  # pyrefly: ignore [bad-argument-type]
                indices,
                self.softmax_scale,
            )
            outputs.append(result.squeeze(0))  # pyrefly: ignore [missing-attribute]

        output = torch.cat(outputs, dim=0).to(query.dtype)
        return output.reshape(metadata.batch_size, metadata.seq_len, *output.shape[1:])
