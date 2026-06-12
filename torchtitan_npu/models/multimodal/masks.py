# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

PADDED_PATCH_COORDINATE = -1


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Build a dense lower-triangular mask using SDPA boolean semantics."""
    positions = torch.arange(seq_len, device=device)
    return positions[:, None] >= positions[None, :]


def build_document_ids(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Assign each token to a packed document separated by EOS tokens."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, seq_len)")

    eos = tokens == eos_id
    doc_ids = eos.cumsum(dim=1)
    return doc_ids - eos.to(doc_ids.dtype)


def build_text_document_causal_mask(
    tokens: torch.Tensor,
    eos_id: int,
) -> torch.Tensor:
    """Build a causal mask that prevents attention across packed documents."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, seq_len)")

    batch, seq_len = tokens.shape
    causal = build_causal_mask(seq_len, tokens.device)
    doc_ids = build_document_ids(tokens, eos_id)
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    return causal.expand(batch, seq_len, seq_len) & same_doc


def build_valid_patch_mask(grid_hw: torch.Tensor) -> torch.Tensor:
    """Return valid image patch positions from zero-padded spatial coordinates.

    ``grid_hw`` must have shape ``(batch, seq_len, 2)``. Positions whose height
    or width coordinate equals ``PADDED_PATCH_COORDINATE`` are padding. The
    returned mask has shape ``(batch, seq_len)``.
    """
    if grid_hw.ndim != 3 or grid_hw.shape[-1] != 2:
        raise ValueError("grid_hw must have shape (batch, seq_len, 2)")

    return (grid_hw != PADDED_PATCH_COORDINATE).all(dim=-1)


def build_encoder_causal_mask(
    valid_tokens: torch.Tensor,
    *,
    allow_padding_self_attention: bool = True,
) -> torch.Tensor:
    """Build a causal encoder mask over valid image patch positions."""
    if valid_tokens.ndim != 2:
        raise ValueError("valid_tokens must have shape (batch, seq_len)")

    valid_tokens = valid_tokens.to(dtype=torch.bool)
    batch, seq_len = valid_tokens.shape
    causal = build_causal_mask(seq_len, valid_tokens.device)
    valid_pair = valid_tokens[:, :, None] & valid_tokens[:, None, :]
    mask = causal.expand(batch, seq_len, seq_len) & valid_pair

    if not allow_padding_self_attention:
        return mask

    # SDPA-specific workaround: dense SDPA returns NaN for queries whose entire
    # row is masked out. Padding rows are ignored downstream, so make them
    # finite with a self row. Other attention kernels may not need this.
    diagonal = torch.eye(seq_len, dtype=torch.bool, device=valid_tokens.device)
    return mask | (~valid_tokens[:, :, None] & diagonal)


def build_encoder_full_mask(
    valid_tokens: torch.Tensor,
    *,
    allow_padding_self_attention: bool = True,
) -> torch.Tensor:
    """Build a full encoder mask over valid image patch positions."""
    if valid_tokens.ndim != 2:
        raise ValueError("valid_tokens must have shape (batch, seq_len)")

    valid_tokens = valid_tokens.to(dtype=torch.bool)
    _batch, seq_len = valid_tokens.shape
    mask = valid_tokens[:, :, None] & valid_tokens[:, None, :]

    if not allow_padding_self_attention:
        return mask

    diagonal = torch.eye(seq_len, dtype=torch.bool, device=valid_tokens.device)
    return mask | (~valid_tokens[:, :, None] & diagonal)
