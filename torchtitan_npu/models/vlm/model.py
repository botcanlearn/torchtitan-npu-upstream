# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast

import torch
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.experiments.vlm.model.model import (
    Llama3Siglip2Transformer,
    SpecialTokens,
)
from torchtitan.models.common.attention import AttentionMasksType

from torchtitan_npu.models.multimodal import (
    DenseMaskSDPA,
    build_config,
    build_encoder_causal_mask,
    build_text_document_causal_mask,
    build_valid_patch_mask,
    config_to_dict,
    require_config,
    scatter_visual_embeddings,
)

from .siglip2 import VisionTransformerNpu, to_npu_vision_transformer_config


def _grid_hw_from_grid_thw(grid_thw: torch.Tensor) -> torch.Tensor:
    return grid_thw[:, :, 1:]


class Llama3Siglip2TransformerNpu(Llama3Siglip2Transformer):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Siglip2Transformer.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # Upstream constructs a concrete VisionTransformer instead of using
        # config.encoder.build(), so replace only the encoder with the NPU
        # subclass while keeping the rest of upstream initialization intact.
        self.encoder = VisionTransformerNpu(require_config(config.encoder, VisionTransformerNpu.Config, "encoder"))

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        if extra_inputs is None or "grid_thw" not in extra_inputs:
            raise ValueError("VLM dense masks require extra_inputs['grid_thw']")
        if tokenizer.eos_id is None:
            raise ValueError("VLM dense masks require tokenizer.eos_id")

        grid_hw = _grid_hw_from_grid_thw(extra_inputs["grid_thw"])
        valid_patches = build_valid_patch_mask(grid_hw)
        masks = {
            "llama3_masks": build_text_document_causal_mask(input_batch, tokenizer.eos_id),
            "encoder_masks": build_encoder_causal_mask(valid_patches),
            "pixel_masks": valid_patches,
        }
        return cast("Any", masks)

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        special_tokens: SpecialTokens,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        hidden_states = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if self.encoder is not None:
            assert attention_masks is not None, "encoder requires attention masks when using VLM NPU dense masks."
            dense_masks = cast("dict[str, torch.Tensor]", attention_masks)
            grid_hw = _grid_hw_from_grid_thw(grid_thw)
            pixel_masks = dense_masks.get("pixel_masks")
            if pixel_masks is None:
                raise ValueError("VLM dense masks require attention_masks['pixel_masks']")
            visual_embeddings = self.encoder(
                pixel_values,
                pixel_masks,
                grid_hw,
                dense_masks["encoder_masks"],
            )
            visual_embeddings = self.projector(visual_embeddings)
            hidden_states = scatter_visual_embeddings(
                hidden_states,
                tokens,
                visual_embeddings,
                pixel_masks,
                special_tokens.img_id,
            )

        for layer in self.layers.values():
            hidden_states = layer(
                hidden_states,
                self.freqs_cis,
                cast("dict[str, torch.Tensor]", attention_masks)["llama3_masks"],
                positions,
            )

        hidden_states = self.norm(hidden_states) if self.norm else hidden_states
        return self.output(hidden_states) if self.output else hidden_states


def to_npu_vlm_config(
    config: Llama3Siglip2Transformer.Config,
) -> Llama3Siglip2TransformerNpu.Config:
    layers = []
    for layer in config.layers:
        attention = layer.attention
        npu_attention = build_config(
            type(attention),
            {
                **config_to_dict(attention),
                "inner_attention": DenseMaskSDPA.Config(),
            },
        )
        layers.append(
            build_config(
                type(layer),
                {
                    **config_to_dict(layer),
                    "attention": npu_attention,
                },
            )
        )
    values = {
        **config_to_dict(config),
        "layers": layers,
        "encoder": to_npu_vision_transformer_config(config.encoder),
    }
    return build_config(Llama3Siglip2TransformerNpu.Config, values)
