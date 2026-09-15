# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed compression metadata for DeepSeek-V4.1 (CP1-only).

Derived from the DSV4 metadata contract; V4.1 runs CP1 exclusively, so the
context-parallel plan fields (exchange routing, container packing under CP,
compressed-level gather) are intentionally absent.  ``build_kernel_layout``
derives the per-document plans from the document boundaries alone: each
document contributes its complete leading blocks, gathered contiguously;
the ``len % ratio`` tail produces no entry and never crosses documents.

The container grid is ``[1, S]`` (packed stream, ``local_batch_size == 1``;
raise ``seq_len`` instead of ``local_batch_size``), so ``batch_size == 1``
and ``seq_len`` is the total token count ``cu_seq_q[-1]``.
"""

from dataclasses import dataclass, fields
from typing import Any, cast

import torch
from torchtitan.models.common.attention import VarlenMetadata

__all__ = [
    "CompressedBlockLayout",
    "CompressedVarlenMetadata",
    "build_compressed_varlen_metadata",
    "build_kernel_layout",
]


@dataclass(kw_only=True, slots=True)
class CompressedBlockLayout:
    """Kernel contract for one compression ratio (the key of ``plans``).

    All tensors are built once per batch.  For ``ratio <= 1`` the layout is
    the empty placeholder: ratio-0 layers are window-only and ratio-1 KV is
    materialized token-for-token, so neither owns compressed blocks.
    """

    cu_seqlens_cmp_k: torch.Tensor | None = None
    """Cumulative compressed-block lengths over the packed stream (int32)."""

    n_cmp_blocks_host: int | None = None
    """Host-cached total compressed block count."""

    block_remainder: torch.Tensor | None
    """Per-sequence incomplete-block remainder; ``None`` for ratio-1 plans."""

    gather_indices: torch.Tensor | None
    """Pooled block-row indices (the leading complete blocks per document)."""

    block_positions: torch.Tensor | None = None
    """Document-relative block-start positions for compressed-key RoPE."""

    first_indices: torch.Tensor | None = None
    """Document-first block ids used by overlap masking."""

    out_width: int | None = None
    """Container grid width (``seq_len // ratio``, the uniform pad target)."""


@dataclass(kw_only=True, slots=True)
class CompressedVarlenMetadata:
    """The DeepSeek-V4.1 varlen attention contract."""

    varlen: "VarlenMetadata"
    """Token-stream boundaries (a plain packed stream: cu_seq_q == cu_seq_k)."""

    plans: dict[int, CompressedBlockLayout]
    """Kernel contract for each ratio present in the model."""

    seq_len_host: int | None = None
    """Host-cached total token count."""

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def seq_len(self) -> int:
        if self.seq_len_host is not None:
            return self.seq_len_host
        return int(self.varlen.cu_seq_q[-1].item())


def build_kernel_layout(
    varlen: VarlenMetadata,
    compress_ratios: tuple[int, ...] | list[int],
) -> dict[int, CompressedBlockLayout]:
    """Build the kernel-contract tier for a plain packed stream."""
    if not hasattr(varlen, "cu_seq_q"):
        raise TypeError(f"build_kernel_layout expects a varlen stream, got {type(varlen)}.")

    cu_seq_q = varlen.cu_seq_q
    if int(cu_seq_q[0].item()) != 0:
        raise ValueError(f"varlen stream must start at token 0, got cu_seq_q[0]={cu_seq_q[0]}.")
    if not torch.equal(cu_seq_q, varlen.cu_seq_k):
        raise ValueError("build_kernel_layout requires a plain stream (cu_seq_q == cu_seq_k).")
    seq_len = int(cu_seq_q[-1].item())

    cu = cu_seq_q.cpu().tolist()
    lengths = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    distinct_ratios = sorted({int(r) for r in compress_ratios})
    device = cu_seq_q.device
    plans: dict[int, CompressedBlockLayout] = {}
    for ratio in distinct_ratios:
        if ratio <= 1:
            plans[ratio] = CompressedBlockLayout(
                cu_seqlens_cmp_k=None,
                block_remainder=None,
                gather_indices=None,
            )
            continue
        c_lens = [length // ratio for length in lengths]
        cu_seqs = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=device),
                torch.tensor(c_lens, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32),
            ]
        )
        pieces = [
            torch.arange(k_start, k_start + ratio * cnt, dtype=torch.int64, device=device)
            for k_start, cnt in zip(cu[:-1], c_lens, strict=True)
            if cnt
        ]
        positions = [torch.arange(0, ratio * cnt, ratio, dtype=torch.int32, device=device) for cnt in c_lens if cnt]
        gather = torch.cat(pieces, dim=0) if pieces else torch.empty((0,), dtype=torch.int64, device=device)
        block_positions = (
            torch.cat(positions, dim=0) if positions else torch.empty((0,), dtype=torch.int32, device=device)
        )
        first_indices = cu_seqs[:-1][torch.diff(cu_seqs) > 0].to(torch.int64)
        plans[ratio] = CompressedBlockLayout(
            cu_seqlens_cmp_k=cu_seqs,
            n_cmp_blocks_host=sum(length // ratio for length in lengths),
            block_remainder=torch.tensor(
                [length % ratio for length in lengths],
                dtype=torch.int32,
                device=device,
            ),
            gather_indices=gather,
            block_positions=block_positions,
            first_indices=first_indices,
            out_width=seq_len // ratio,
        )
    return plans


def build_compressed_varlen_metadata(
    varlen: VarlenMetadata,
    compress_ratios: tuple[int, ...] | list[int],
) -> CompressedVarlenMetadata:
    """Build the V4.1 varlen contract for the rank-local token stream."""
    plans = build_kernel_layout(varlen, compress_ratios)
    return CompressedVarlenMetadata(
        varlen=varlen,
        plans=plans,
        seq_len_host=int(varlen.cu_seq_q[-1].item()),
    )


def register_pytree_node_for_dataclass(cls: type) -> None:
    """Register a kw-only dataclass as a pytree node (idempotent)."""
    from torch.utils._pytree import SUPPORTED_NODES, GetAttrKey, KeyEntry, register_pytree_node

    if cls in SUPPORTED_NODES:
        return
    field_names = [f.name for f in fields(cls)]

    def flatten(obj):
        return [getattr(obj, name) for name in field_names], None

    def flatten_with_keys(obj) -> tuple[list[tuple[KeyEntry, Any]], None]:
        keys = cast(
            "list[tuple[KeyEntry, Any]]",
            [(GetAttrKey(name), getattr(obj, name)) for name in field_names],
        )
        return keys, None

    def unflatten(values, context):
        return cls(**dict(zip(field_names, values, strict=True)))

    register_pytree_node(
        cls,
        flatten,
        unflatten,
        flatten_with_keys_fn=flatten_with_keys,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
    )


register_pytree_node_for_dataclass(CompressedBlockLayout)
register_pytree_node_for_dataclass(CompressedVarlenMetadata)
