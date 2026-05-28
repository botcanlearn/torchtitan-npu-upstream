# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from dataclasses import dataclass

import einops as E
import torch
import torch.nn.functional as F

from torchtitan.experiments.vlm.model import siglip2 as upstream_siglip2
from torchtitan.models.common.attention import (
    AttentionMasksType,
    LocalMapInnerAttention,
)
from torchtitan.protocols.module import Module, ModuleDict

from torchtitan_npu.models.multimodal import (
    build_config,
    config_to_dict,
    DenseMaskSDPA,
    require_config,
)


def resize_positional_embeddings_zero_padded(
    pos_embeddings: torch.Tensor,
    spatial_shapes: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Resize positional embeddings and leave padded image rows as zeros."""
    resized_embeddings = pos_embeddings.new_zeros(
        (spatial_shapes.shape[0], max_length, pos_embeddings.shape[-1])
    )

    pos_embedding_channels_first = E.rearrange(pos_embeddings, "h w d -> 1 d h w")
    shape_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, shape in enumerate(spatial_shapes.tolist()):
        height, width = (int(shape[0]), int(shape[1]))
        if height <= 0 or width <= 0:
            continue
        shape_to_indices[(height, width)].append(index)

    for (height, width), indices in shape_to_indices.items():
        resized_emb = F.interpolate(
            pos_embedding_channels_first,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        resized_embeddings[indices, : height * width] = E.rearrange(
            resized_emb, "1 d h w -> (h w) d"
        )

    return resized_embeddings


class VisionEmbeddingsNpu(upstream_siglip2.VisionEmbeddings):
    @dataclass(kw_only=True, slots=True)
    class Config(upstream_siglip2.VisionEmbeddings.Config):
        pass

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self, pixels: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        patch_embeddings = self.patch_embedding(pixels)

        pos_embeddings = self.position_embedding.weight.reshape(
            self.n_pos_embs, self.n_pos_embs, -1
        )
        spatial_h = E.reduce(grid_hw[:, :, 0], "n l -> n", reduction="max") + 1
        spatial_w = E.reduce(grid_hw[:, :, 1], "n l -> n", reduction="max") + 1
        spatial_shapes = torch.stack([spatial_h, spatial_w], dim=-1).long()
        resized_positional_embeddings = resize_positional_embeddings_zero_padded(
            pos_embeddings,
            spatial_shapes,
            max_length=pixels.shape[1],
        )
        return patch_embeddings + resized_positional_embeddings


class VisionAttentionNpu(upstream_siglip2.Attention):
    @dataclass(kw_only=True, slots=True)
    class Config(upstream_siglip2.Attention.Config):
        # ``to_npu_vision_attention_config`` injects DenseMaskSDPA.Config here.
        # Keep the type at the common inner-attention boundary so this config
        # can still accept compatible multimodal attention backends later.
        inner_attention: LocalMapInnerAttention.Config

    def __init__(self, config: Config):
        # Mirrors torchtitan/experiments/vlm/model/siglip2.py:
        # Attention.__init__. Upstream currently hard-codes
        # FlexAttention.Config().build() instead of reading config.inner_attention,
        # so NPU must build the same fields manually and then use the
        # configurable inner attention. Keep dim/head_dim, q/k/v/out projections,
        # and inner_attention in sync with that upstream constructor.
        Module.__init__(self)
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.q_proj = config.qkv_proj.build()
        self.k_proj = config.qkv_proj.build()
        self.v_proj = config.qkv_proj.build()
        self.out_proj = config.out_proj.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
    ) -> torch.Tensor:
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = E.rearrange(xq, "b l (h d) -> b l h d", d=self.head_dim)
        xk = E.rearrange(xk, "b l (h d) -> b l h d", d=self.head_dim)
        xv = E.rearrange(xv, "b l (h d) -> b l h d", d=self.head_dim)

        output = self.inner_attention(xq, xk, xv, attention_masks=attention_masks)
        output = E.rearrange(output, "b l h d -> b l (h d)").contiguous()
        return self.out_proj(output)


class VisionTransformerLayerNpu(upstream_siglip2.TransformerLayer):
    @dataclass(kw_only=True, slots=True)
    class Config(upstream_siglip2.TransformerLayer.Config):
        pass

    def __init__(self, config: Config):
        # Mirrors torchtitan/experiments/vlm/model/siglip2.py:
        # TransformerLayer.__init__. Upstream hard-codes Attention(config.self_attn),
        # which builds FlexAttention before we replace it. Build the same layer
        # fields manually and use VisionAttentionNpu directly.
        Module.__init__(self)
        self.layer_norm1 = upstream_siglip2.LayerNorm(
            config.dim, eps=config.layer_norm_eps
        )
        self.self_attn = VisionAttentionNpu(
            require_config(config.self_attn, VisionAttentionNpu.Config, "self_attn")
        )
        self.layer_norm2 = upstream_siglip2.LayerNorm(
            config.dim, eps=config.layer_norm_eps
        )
        self.mlp = upstream_siglip2.FeedForward(config.mlp)


class VisionTransformerNpu(upstream_siglip2.VisionTransformer):
    @dataclass(kw_only=True, slots=True)
    class Config(upstream_siglip2.VisionTransformer.Config):
        pass

    def __init__(self, config: Config):
        # Mirrors torchtitan/experiments/vlm/model/siglip2.py:
        # VisionTransformer.__init__. Build the same fields manually so NPU does
        # not allocate upstream VisionEmbeddings/TransformerLayer/FlexAttention
        # objects only to replace them.
        Module.__init__(self)
        self.attn_mask_type = config.attn_mask_type
        # Match upstream VisionTransformer.__init__ hard-coded eos_id.
        self.eos_id = 11
        self.embeddings = VisionEmbeddingsNpu(config.embeddings)
        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.layers):
            self.layers[str(i)] = VisionTransformerLayerNpu(
                require_config(
                    layer_config,
                    VisionTransformerLayerNpu.Config,
                    f"layers[{i}]",
                )
            )
        self.post_layernorm = upstream_siglip2.LayerNorm(
            config.dim, eps=config.layer_norm_eps
        )


def to_npu_vision_embeddings_config(
    config: upstream_siglip2.VisionEmbeddings.Config,
) -> upstream_siglip2.VisionEmbeddings.Config:
    return build_config(VisionEmbeddingsNpu.Config, config_to_dict(config))


def to_npu_vision_attention_config(
    config: upstream_siglip2.Attention.Config,
) -> upstream_siglip2.Attention.Config:
    return build_config(
        VisionAttentionNpu.Config,
        {
            **config_to_dict(config),
            "inner_attention": DenseMaskSDPA.Config(),
        },
    )


def to_npu_vision_layer_config(
    config: upstream_siglip2.TransformerLayer.Config,
) -> upstream_siglip2.TransformerLayer.Config:
    return build_config(
        VisionTransformerLayerNpu.Config,
        {
            **config_to_dict(config),
            "self_attn": to_npu_vision_attention_config(config.self_attn),
        },
    )


def to_npu_vision_transformer_config(
    config: upstream_siglip2.VisionTransformer.Config,
) -> upstream_siglip2.VisionTransformer.Config:
    return build_config(
        VisionTransformerNpu.Config,
        {
            **config_to_dict(config),
            "embeddings": to_npu_vision_embeddings_config(config.embeddings),
            "layers": [to_npu_vision_layer_config(layer) for layer in config.layers],
        },
    )
