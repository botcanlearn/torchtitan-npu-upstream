# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The text/multimodal split: which half owns what, and what each half rejects.

V4.1 has two models, not one model with an optional tower: the text stack
(:class:`DeepSeekV41Model`) carries no vision parameter and no modality input, and the
multimodal stack (:class:`DeepSeekV41MultimodalModel`) subclasses it with the tower, the
markers, the router's vision bias and the context-parallel rejection.  These tests pin
that partition, because the whole point of the split is that each half's parameter set
and routing behaviour is readable from its class.
"""

import dataclasses
import importlib
import inspect

import pytest

from torchtitan_npu.models.deepseek_v4_1.model import (
    DeepSeekV41Model,
    DeepSeekV41MultimodalModel,
    DeepSeekV41TransformerBlock,
)


@pytest.fixture(scope="module")
def registry():
    return importlib.import_module("torchtitan_npu.models.deepseek_v4_1")


@pytest.fixture(scope="module")
def recipes():
    return importlib.import_module("torchtitan_npu.models.deepseek_v4_1.config_registry")


def _flavors(registry):
    return {
        "text": registry.deepseek_v4_1_debugmodel_text_config(),
        "multimodal": registry.deepseek_v4_1_debugmodel_config(),
    }


def test_the_two_flavors_build_the_two_classes(registry):
    """The flavor's class is the modality: there is no half-built middle case."""
    text, multimodal = _flavors(registry).values()
    assert type(text) is DeepSeekV41Model.Config
    assert type(multimodal) is DeepSeekV41MultimodalModel.Config
    assert isinstance(multimodal, DeepSeekV41Model.Config)


def test_only_the_multimodal_half_carries_vision_parameters(registry):
    """`bias_vl` and the vision widths are the vision half's, not the text half's.

    The block itself declares no modality: it takes an ``image_mask`` and defaults it to
    ``None``, which is what a text-only forward passes, so a per-layer enable flag would
    be a second way to say what the argument already says.
    """
    text, multimodal = _flavors(registry).values()
    assert not any(layer.moe.router.vision_enabled for layer in text.layers)
    assert all(layer.moe.router.vision_enabled for layer in multimodal.layers)
    assert "vision_enabled" not in {field.name for field in dataclasses.fields(DeepSeekV41TransformerBlock.Config)}
    # A text-only width set has no vision widths to size, and the factory refuses a
    # width set that disagrees with the modality.
    assert (registry._DEBUG_TEXT_WIDTHS.vision_dim, registry._DEBUG_TEXT_WIDTHS.vision_heads) == (0, 0)
    assert registry._DEBUG_WIDTHS.vision_dim > 0

    with pytest.raises(ValueError, match="needs no vision widths"):
        registry._make_v41_config(
            n_layers=40,
            compress_ratios=registry.V41_FULL_COMPRESS_RATIOS,
            kv_source_layers=registry.V41_KV_SOURCE_LAYERS,
            index_source_layers=registry.V41_FULL_INDEX_SOURCE_LAYERS,
            candidate_source_layer=registry.V41_CANDIDATE_SOURCE_LAYER,
            moe_comm_backend="standard",
            non_blocking_capacity_factor=None,
            vision=False,
            widths=registry._DEBUG_WIDTHS,
        )


def test_moe_is_called_without_input_ids():
    """No V4.1 layer hashes, so the block hands the MoE no token ids.

    ``input_ids`` stays in ``HashMoE.forward`` for DSV4's hash layers, but the V4.1 call
    site must not pass it: the router reads it only on the hash path, which no V4.1
    layer builds.
    """
    source = inspect.getsource(DeepSeekV41TransformerBlock.forward)
    assert "self.moe(" in source
    moe_call = source[source.index("self.moe(") : source.index("\n", source.index("self.moe("))]
    assert "input_ids" not in moe_call, moe_call
    assert "image_mask" in moe_call, moe_call
    # Engram still needs the ids: it hashes the token n-grams.
    assert "input_ids" in source[: source.index("self.moe(")], source[: source.index("self.moe(")]


def test_context_parallel_has_exactly_one_declaration_site(registry):
    """The CP gate is declared once per stack, on the config -- its only reader.

    ``update_from_config`` runs before any module exists, so the config is the only thing
    that can be asked; a second declaration on the model class would be a copy nothing
    reads and everything can drift from.
    """
    assert DeepSeekV41Model.Config.accepts_context_parallel is True
    assert DeepSeekV41MultimodalModel.Config.accepts_context_parallel is False
    assert DeepSeekV41Model.Config.accepts_context_parallel is not (
        DeepSeekV41MultimodalModel.Config.accepts_context_parallel
    )
    for cls in (DeepSeekV41Model, DeepSeekV41MultimodalModel):
        assert "accepts_context_parallel" not in vars(cls), cls
    assert issubclass(DeepSeekV41MultimodalModel, DeepSeekV41Model)


def _with_cp(trainer, degree: int):
    trainer.training = dataclasses.replace(trainer.training, global_batch_size=8)
    trainer.parallelism = dataclasses.replace(trainer.parallelism, context_parallel_degree=degree)
    return trainer


def test_multimodal_rejects_context_parallel_and_text_accepts_it(registry, recipes):
    """CP is the text half's; the vision half keeps rejecting it until it has a story."""
    multimodal = _with_cp(recipes.deepseek_v4_1_debugmodel_multimodal(), 2)
    with pytest.raises(NotImplementedError, match="multimodal stack"):
        multimodal.model_spec.model.update_from_config(config=multimodal)

    # The text half gets past the gate: whatever else CP needs is step 3's, and the gate
    # itself must not be it.
    text = _with_cp(recipes.deepseek_v4_1_debugmodel_text(), 2)
    try:
        text.model_spec.model.update_from_config(config=text)
    except ValueError:
        # Downstream of the gate and part of the CP work itself (the sharding policy has
        # no CP placement yet); the gate is what this test is about.
        pass


def test_text_recipe_builds_the_text_stack_and_loader(recipes):
    """The recipe pairs the text model with the text loader and its alignment."""
    trainer = recipes.deepseek_v4_1_debugmodel_text()
    assert isinstance(trainer.model_spec.model, DeepSeekV41Model.Config)
    assert not isinstance(trainer.model_spec.model, DeepSeekV41MultimodalModel.Config)
    assert type(trainer.dataloader).__name__ == "Config"
    assert trainer.dataloader.per_doc_alignment == 2

    multimodal = recipes.deepseek_v4_1_debugmodel_multimodal()
    assert isinstance(multimodal.model_spec.model, DeepSeekV41MultimodalModel.Config)
    assert multimodal.dataloader.per_doc_alignment == 2


@pytest.mark.parametrize(
    ("flavor", "multimodal"),
    [
        ("deepseek_v4_1_debugmodel", True),
        ("deepseek_v4_1_debugmodel_text", False),
        ("deepseek_v4_1_flash_40layers_16experts_vision", True),
        ("deepseek_v4_1_flash_40layers_16experts_text", False),
    ],
)
def test_every_registered_flavor_declares_its_modality(registry, flavor, multimodal):
    """A flavor name and the class it builds cannot drift apart silently."""
    assert flavor in registry.deepseek_v4_1_configs, sorted(registry.deepseek_v4_1_configs)
    config = registry.deepseek_v4_1_configs[flavor](non_blocking_capacity_factor=None)
    assert isinstance(config, DeepSeekV41MultimodalModel.Config) is multimodal, flavor
    assert ("_text" in flavor) is not multimodal, flavor
