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

import contextlib
import dataclasses
import importlib
import inspect

import pytest
import torch.nn as nn

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
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
    with contextlib.suppress(ValueError):
        # Downstream of the gate and part of the CP work itself (the sharding policy has
        # no CP placement yet); the gate is what this test is about.
        text.model_spec.model.update_from_config(config=text)


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


class _RecordedHooks(nn.Module):
    """Stands in for a text model during ``parallelize``.

    Deliberately *not* a :class:`DeepSeekV41MultimodalModel`: a text stack cannot satisfy
    the multimodal branch, so the hooks must never be reached.  An ``nn.Module`` because
    the sharding validation walks the model it is handed.
    """

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []
        # The sharding validation walks the decoder layers, so the stand-in needs the
        # attribute even though it has none of its own.
        self.layers = nn.ModuleDict()

    def apply_activation_checkpointing_extensions(self, policy) -> None:
        self.calls.append("ac")

    def apply_fsdp_extensions(self, **kwargs) -> None:
        self.calls.append("fsdp")


@pytest.fixture
def parallelize_deps(monkeypatch):
    """Neutralize the distributed half of ``parallelize_deepseek_v4_1``.

    The model-side dispatch is what is under test, so everything needing a real process
    group is replaced; ``isinstance`` on the model class is what decides the hooks.
    """
    from unittest.mock import Mock

    from torchtitan_npu.models.deepseek_v4_1 import parallelize as par

    # ``apply_activation_checkpointing`` is left real: the policy walk is cheap on a CPU
    # model, and it is the function whose dispatch is under test.
    for name in ("_shard_engram_tables", "apply_fsdp_to_decoder"):
        monkeypatch.setattr(par, name, Mock(name=name, return_value=set()))
    parallel_dims = Mock()
    parallel_dims.ep_enabled = False
    parallel_dims.pp_enabled = False
    parallel_dims.get_mesh.return_value = Mock()
    return par, parallel_dims


def _parallelize(par, parallel_dims, recipes, model, *, vision: bool):
    trainer = recipes.deepseek_v4_1_debugmodel_multimodal() if vision else recipes.deepseek_v4_1_debugmodel_text()
    # The dispatch under test is backend-independent; the recipe's spmd_types path routes
    # through the sharding validation and the model's own parallelize instead, so this
    # takes the plain FSDP backend.
    parallelism = dataclasses.replace(trainer.parallelism, spmd_backend="partial_dtensor")
    return par.parallelize_deepseek_v4_1(
        model,
        parallel_dims=parallel_dims,
        training=trainer.training,
        parallelism=parallelism,
        compile_config=trainer.compile,
        ac_config=trainer.activation_checkpoint,
        dump_folder=".",
    )


def test_the_extension_hooks_live_only_on_the_multimodal_stack():
    """The text stack must not carry a hook nobody would call for it."""
    assert not hasattr(DeepSeekV41Model, "apply_activation_checkpointing_extensions")
    assert not hasattr(DeepSeekV41Model, "apply_fsdp_extensions")
    assert hasattr(DeepSeekV41MultimodalModel, "apply_activation_checkpointing_extensions")
    assert hasattr(DeepSeekV41MultimodalModel, "apply_fsdp_extensions")


def test_parallelize_never_asks_the_text_stack_for_the_hooks(registry, recipes, parallelize_deps):
    """A text run reaches both dispatch points without the hooks existing.

    A stand-in that is deliberately *not* the multimodal class exercises the same branch
    a real text model takes; if the dispatch regressed to an unconditional call this
    raises ``AttributeError`` instead of passing.
    """
    par, parallel_dims = parallelize_deps
    model = _RecordedHooks()
    _parallelize(par, parallel_dims, recipes, model, vision=False)
    assert model.calls == []


def test_parallelize_asks_the_multimodal_stack_for_both_hooks(registry, recipes, parallelize_deps):
    """The multimodal branch reaches both hooks, which is the only reason they exist."""
    par, parallel_dims = parallelize_deps
    model = build_cpu_model(registry.deepseek_v4_1_debugmodel_config())
    calls: list[str] = []
    # The hooks are this class's own methods; recording them here keeps the assertion
    # about dispatch, while ``test_multimodal_ac_hook_wraps_every_vision_block`` covers
    # what the AC hook does with a real policy.
    model.apply_activation_checkpointing_extensions = lambda policy: calls.append("ac")
    model.apply_fsdp_extensions = lambda **kwargs: calls.append("fsdp")
    _parallelize(par, parallel_dims, recipes, model, vision=True)
    assert calls == ["ac", "fsdp"]


def test_multimodal_ac_hook_wraps_every_vision_block(registry):
    """The AC hook is the tower's only route into the selected policy.

    The policy's own walk covers decoder layers; a vision block lives outside that list,
    so it is wrapped here or not at all.
    """
    config = registry.deepseek_v4_1_debugmodel_config()
    model = build_cpu_model(config)
    wrapped: list[str] = []

    class _Policy:
        @staticmethod
        def _wrap_block(block, *, base_fqn):
            wrapped.append(base_fqn)
            return block

    model.apply_activation_checkpointing_extensions(_Policy())
    expected = [f"vision_encoder.blocks.{name}" for name, _ in model.vision_encoder.blocks.named_children()]
    assert wrapped == expected
    assert len(wrapped) == 32
