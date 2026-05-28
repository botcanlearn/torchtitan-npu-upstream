# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def scatter_visual_embeddings(
    hidden_states: torch.Tensor,
    tokens: torch.Tensor,
    visual_embeddings: torch.Tensor,
    visual_token_mask: torch.Tensor,
    image_token_id: int,
    *,
    validate_token_count: bool = True,
) -> torch.Tensor:
    """Replace image placeholder slots in ``hidden_states`` with visual embeddings.

    Shape contract:
      - ``hidden_states``: ``(batch, seq_len, dim)``
      - ``tokens``: ``(batch, seq_len)``
      - ``visual_embeddings``: ``(num_images, image_seq_len, dim)``
      - ``visual_token_mask``: ``(num_images, image_seq_len)`` with dtype ``torch.bool``
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape (batch, seq_len, dim), got {tuple(hidden_states.shape)}"
        )
    if tokens.shape != hidden_states.shape[:2]:
        raise ValueError(
            "tokens must have shape matching hidden_states leading dims "
            f"{tuple(hidden_states.shape[:2])}, got {tuple(tokens.shape)}"
        )
    if visual_embeddings.ndim != 3:
        raise ValueError(
            "visual_embeddings must have shape (num_images, image_seq_len, dim), "
            f"got {tuple(visual_embeddings.shape)}"
        )
    if visual_token_mask.shape != visual_embeddings.shape[:2]:
        raise ValueError(
            "visual_token_mask must match visual_embeddings leading dims "
            f"{tuple(visual_embeddings.shape[:2])}, got {tuple(visual_token_mask.shape)}"
        )
    if visual_token_mask.dtype is not torch.bool:
        raise ValueError(
            f"visual_token_mask must have dtype torch.bool, got {visual_token_mask.dtype}"
        )
    if visual_embeddings.shape[-1] != hidden_states.shape[-1]:
        raise ValueError(
            "visual_embeddings hidden dim must match hidden_states hidden dim, "
            f"got visual_embeddings {tuple(visual_embeddings.shape)} and "
            f"hidden_states {tuple(hidden_states.shape)}"
        )

    _, _, dim = hidden_states.shape
    image_slots = (tokens == image_token_id).unsqueeze(-1).to(hidden_states.device)
    visual_source = visual_embeddings.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    visual_mask = visual_token_mask.to(device=hidden_states.device).unsqueeze(-1)
    visual_flatten = torch.masked_select(
        visual_source,
        mask=visual_mask,
    )

    if validate_token_count:
        num_visual_embeddings = visual_flatten.numel() // dim
        num_image_slots = int(image_slots.sum().item())
        if num_visual_embeddings != num_image_slots:
            raise ValueError(
                f"Different number of visual embeddings {num_visual_embeddings} "
                f"with placeholder in input token embeddings {num_image_slots}"
            )

    hidden_states.masked_scatter_(mask=image_slots, source=visual_flatten)
    return hidden_states
