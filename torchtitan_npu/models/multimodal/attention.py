# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import FlexAttention, LocalMapInnerAttention


class DenseMaskSDPA(LocalMapInnerAttention):
    """SDPA attention for NPU multimodal models with dense boolean masks.

    The mask follows PyTorch SDPA convention: True means this query/key element
    is allowed to attend. Inputs use TorchTitan attention layout `(B, S, H, D)`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        """Advertise mask-needing attention to Trainer.

        Trainer only calls ``model.get_attention_masks`` for Flex/Varlen config
        types. VLM NPU overrides that method to return dense boolean masks, so
        this inherits FlexAttention.Config for dispatch only; the runtime module
        remains DenseMaskSDPA.
        """

        pass

    def __init__(self, config: Config | None = None) -> None:
        super().__init__(config or DenseMaskSDPA.Config())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_masks: torch.Tensor | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_mask = None
        if attention_masks is not None:
            attn_mask = attention_masks[:, None, :, :].to(device=q.device)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out.transpose(1, 2)
