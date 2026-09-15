# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 reference-attention metadata tier and its extension.

The golden attention consumes a reference tier on the common metadata: the
per-token document ids and positions, the per-ratio dense attendability
mask and the container-slot coordinates.  The tier is no-CP-shaped (contiguous documents):
``cu_seq_q == cu_seq_k`` is enforced here.

The dense mask has exactly one owner: ``reference.ratios[ratio].dense_mask``
of this tier.  V4.1 materializes ratio 1 as a true token-for-token
shared/global-KV container (``materialized_ratios=(1,)``); ratio 2 keeps
each document's complete leading blocks with no cross-document tail
splicing.
"""

from dataclasses import dataclass
from typing import cast

import torch

from torchtitan_npu.models.common.metadata_extension import MetadataExtension

from .metadata import CompressedBlockLayout, CompressedVarlenMetadata


@dataclass(kw_only=True, slots=True)
class ReferenceRatioLayout:
    """Reference-attention tensors for one compression ratio."""

    dense_mask: torch.Tensor | None = None
    """Boolean attendability over a materialized container grid."""

    doc_of_block: torch.Tensor | None = None
    """Document id of each materialized container slot."""

    block_local: torch.Tensor | None = None
    """Document-local position/block index of each materialized slot."""


@dataclass(kw_only=True, slots=True)
class ReferenceLayout:
    """The V4.1 reference-attention tier."""

    doc_of_token: torch.Tensor
    pos_in_doc: torch.Tensor
    ratios: dict[int, ReferenceRatioLayout]


def _build_dense_mask(
    doc_of_block: torch.Tensor,
    block_local: torch.Tensor,
    doc_of_token: torch.Tensor,
    pos_in_doc: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """Attendability over the container grid: same document and causal."""
    same_doc = doc_of_block.unsqueeze(1) == doc_of_token.unsqueeze(2)
    causal_limit = torch.div(pos_in_doc + 1, ratio, rounding_mode="floor").unsqueeze(2)
    causal = block_local.unsqueeze(1) < causal_limit
    return (same_doc & causal).unsqueeze(1)


def _materialize_identity_ratio(
    *,
    batch_size: int,
    seq_len: int,
    doc_of_token: torch.Tensor,
    pos_in_doc: torch.Tensor,
) -> ReferenceRatioLayout:
    """Materialize ratio-1 as a true token-for-token global-KV container."""
    doc_of_block = doc_of_token.clone()
    block_local = pos_in_doc.clone()
    return ReferenceRatioLayout(
        dense_mask=_build_dense_mask(
            doc_of_block,
            block_local,
            doc_of_token,
            pos_in_doc,
            1,
        ),
        doc_of_block=doc_of_block.view(batch_size, seq_len),
        block_local=block_local.view(batch_size, seq_len),
    )


def derive_reference_layout(
    cu_seq_q: torch.Tensor,
    plans: dict[int, "CompressedBlockLayout"],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    *,
    materialized_ratios: tuple[int, ...] = (),
) -> ReferenceLayout:
    """Build the no-CP reference tier for the configured materialization policy."""
    total_tokens = int(cu_seq_q[-1].item())
    lengths = torch.diff(cu_seq_q).to(torch.int32)
    doc_of_token_flat = torch.repeat_interleave(
        torch.arange(len(lengths), device=device, dtype=torch.int32),
        lengths,
    )
    pos_in_doc_flat = (torch.arange(total_tokens, device=device) - cu_seq_q[doc_of_token_flat.long()]).to(torch.int32)
    doc_of_token = doc_of_token_flat.view(batch_size, seq_len)
    pos_in_doc = pos_in_doc_flat.view(batch_size, seq_len)

    materialized = set(materialized_ratios)
    ratios: dict[int, ReferenceRatioLayout] = {}
    for ratio, plan in plans.items():
        if ratio == 1 and ratio in materialized:
            ratios[ratio] = _materialize_identity_ratio(
                batch_size=batch_size,
                seq_len=seq_len,
                doc_of_token=doc_of_token,
                pos_in_doc=pos_in_doc,
            )
            continue
        if ratio <= 1:
            ratios[ratio] = ReferenceRatioLayout()
            continue

        container_width = seq_len // ratio
        cu_cmp = plan.cu_seqlens_cmp_k
        assert cu_cmp is not None, "materialized ratio > 1 plans must carry cu_seqlens_cmp_k"
        n_blocks = int(cu_cmp[-1].item())
        if n_blocks == 0:
            empty_slots = torch.full(
                (batch_size * container_width,),
                -1,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, container_width)
            doc_of_block = empty_slots
            block_local = empty_slots
            dense_mask = _build_dense_mask(
                empty_slots,
                empty_slots,
                doc_of_token,
                pos_in_doc,
                ratio,
            )
        else:
            bids = torch.arange(n_blocks, device=device, dtype=torch.int64)
            seq_ids = torch.searchsorted(cu_cmp[1:], bids, right=True)
            local_idx = bids - cu_cmp[seq_ids]
            doc_of_block = torch.full(
                (batch_size * container_width,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            block_local = torch.full(
                (batch_size * container_width,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            doc_of_block[:n_blocks] = seq_ids.to(torch.int32)
            block_local[:n_blocks] = local_idx.to(torch.int32)
            dense_mask = _build_dense_mask(
                doc_of_block.view(batch_size, container_width),
                block_local.view(batch_size, container_width),
                doc_of_token,
                pos_in_doc,
                ratio,
            )
        ratios[ratio] = ReferenceRatioLayout(
            dense_mask=dense_mask,
            doc_of_block=doc_of_block.view(batch_size, container_width),
            block_local=block_local.view(batch_size, container_width),
        )

    return ReferenceLayout(
        doc_of_token=doc_of_token,
        pos_in_doc=pos_in_doc,
        ratios=ratios,
    )


@dataclass(kw_only=True, slots=True)
class ReferenceCompressedVarlenMetadata(CompressedVarlenMetadata):
    """The common contract plus the reference tier."""

    reference: ReferenceLayout


class ReferenceMetadataExtension(MetadataExtension):
    """Reference-tier post-process of the common attention metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(MetadataExtension.Config):
        materialized_ratios: tuple[int, ...] = ()
        """Ratios that represent real long-range containers in this model."""

    def __call__(self, metadata) -> ReferenceCompressedVarlenMetadata:
        if not isinstance(metadata, CompressedVarlenMetadata):
            raise TypeError(
                "the reference tier requires the V4.1 common metadata "
                f"(CompressedVarlenMetadata), got {type(metadata)}."
            )
        if not torch.equal(metadata.varlen.cu_seq_q, metadata.varlen.cu_seq_k):
            raise ValueError("the reference tier requires cu_seq_q == cu_seq_k (contiguous documents)")
        cfg = cast("ReferenceMetadataExtension.Config", self.config)
        reference = derive_reference_layout(
            metadata.varlen.cu_seq_q,
            metadata.plans,
            metadata.batch_size,
            metadata.seq_len,
            metadata.varlen.cu_seq_q.device,
            materialized_ratios=cfg.materialized_ratios,
        )
        return ReferenceCompressedVarlenMetadata(
            varlen=metadata.varlen,
            plans=metadata.plans,
            seq_len_host=metadata.seq_len_host,
            reference=reference,
        )
