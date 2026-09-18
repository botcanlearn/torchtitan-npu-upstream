# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtitan.config.override import apply_overrides

from torchtitan_npu.models.common.rope import HalfRotation
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.config_registry import (
    deepseek_v4_1_debugmodel_multimodal,
    deepseek_v4_1_debugmodel_multimodal_a3,
    deepseek_v4_1_flash_40layers_16experts_multimodal_a5,
)
from torchtitan_npu.models.deepseek_v4_1.mhc import HcPre
from torchtitan_npu.override.common.rope import AscComplexRoPE, AscHalfRotation, WorkaroundComplexRoPE
from torchtitan_npu.override.deepseek_v4_1.mhc import AscV41HcPre
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention


@pytest.mark.parametrize(
    "recipe,rope_type,vision_type",
    [
        (deepseek_v4_1_debugmodel_multimodal, WorkaroundComplexRoPE, HalfRotation),
        (deepseek_v4_1_debugmodel_multimodal_a3, AscComplexRoPE, AscHalfRotation),
    ],
)
def test_recipe_selects_shared_rope_for_text_compressor_indexer_and_vision(recipe, rope_type, vision_type):
    trainer = recipe()
    overrides = trainer.override
    cfg = trainer.model_spec.model
    apply_overrides(overrides, cfg)
    assert type(cfg.layers[0].attention.rope) is rope_type.Config
    assert type(cfg.layers[2].attention.compressor.rope) is rope_type.Config
    assert type(cfg.layers[2].attention.indexer.rope) is rope_type.Config
    assert type(cfg.vision_encoder.rotary) is vision_type.Config


def test_asc_workaround_bridge_preserves_the_split_site_contract():
    """The narrow bridge derives from the Workaround config without dropping the
    fields the model's builder set (split prefix width, theta, YaRN, length)."""
    trainer = deepseek_v4_1_debugmodel_multimodal_a3()
    cfg = trainer.model_spec.model
    attention_before = cfg.layers[0].attention.rope
    compress_before = cfg.layers[2].attention.compressor.rope
    apply_overrides(trainer.override, cfg)
    for before, after in (
        (attention_before, cfg.layers[0].attention.rope),
        (compress_before, cfg.layers[2].attention.compressor.rope),
    ):
        assert type(after) is AscComplexRoPE.Config
        assert after.split == before.split
        assert after.dim == before.dim
        assert after.theta == before.theta
        assert after.max_seq_len == before.max_seq_len
        assert after.scaling == before.scaling


def test_a5_recipe_selects_sparse_and_sinkhorn_and_keeps_the_loss_config():
    """The A5 recipe's override list actually replaces the fused targets, and
    the distillation config rides along unchanged."""
    trainer = deepseek_v4_1_flash_40layers_16experts_multimodal_a5()
    cfg = trainer.model_spec.model
    apply_overrides(trainer.override, cfg)

    inner = cfg.layers[2].attention.inner_attention
    assert type(inner) is AscV41SparseAttention.Config
    assert issubclass(AscV41SparseAttention, CompressedSparseInnerAttention2)
    assert inner.aux_loss is not None
    assert inner.aux_loss.coeff == 0.01
    assert inner.aux_loss.softmax_scale == inner.softmax_scale
    assert inner.compress_ratio == cfg.layers[2].attention.compress_ratio

    # field preservation through derive() is covered by test_mhc_adapters;
    # here the recipe must actually select the fused targets
    for layer in cfg.layers[:6]:
        assert type(layer.hc_attn_pre) is AscV41HcPre.Config
        assert issubclass(AscV41HcPre, HcPre)
