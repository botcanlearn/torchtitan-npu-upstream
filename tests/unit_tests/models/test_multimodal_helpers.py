# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.protocols.model_converter import ModelConvertersContainer


def _ensure_package_stub(module_name: str, path: Path) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
    module.__path__ = [str(path)]


def _load_module(module_name: str, module_path: Path):
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "__file__", None) is not None:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _module_symbol(module, name: str):
    return vars(module)[name]


def _load_multimodal_modules():
    repo_root = Path(__file__).resolve().parents[3]
    npu_root = repo_root / "torchtitan_npu"
    models_root = npu_root / "models"
    multimodal_root = models_root / "multimodal"

    _ensure_package_stub("torchtitan_npu", npu_root)
    _ensure_package_stub("torchtitan_npu.models", models_root)
    _ensure_package_stub("torchtitan_npu.models.multimodal", multimodal_root)

    attention = _load_module(
        "torchtitan_npu.models.multimodal.attention",
        multimodal_root / "attention.py",
    )
    masks = _load_module(
        "torchtitan_npu.models.multimodal.masks",
        multimodal_root / "masks.py",
    )
    scatter = _load_module(
        "torchtitan_npu.models.multimodal.scatter",
        multimodal_root / "scatter.py",
    )
    multimodal = _load_module(
        "torchtitan_npu.models.multimodal",
        multimodal_root / "__init__.py",
    )

    return attention, masks, scatter, multimodal


def _load_vlm_config_registry():
    repo_root = Path(__file__).resolve().parents[3]
    npu_root = repo_root / "torchtitan_npu"
    models_root = npu_root / "models"
    vlm_root = models_root / "vlm"

    _ensure_package_stub("torchtitan_npu", npu_root)
    _ensure_package_stub("torchtitan_npu.models", models_root)
    _ensure_package_stub("torchtitan_npu.models.vlm", vlm_root)
    return _load_module(
        "torchtitan_npu.models.vlm.config_registry",
        vlm_root / "config_registry.py",
    )


def _load_vlm_model_module():
    repo_root = Path(__file__).resolve().parents[3]
    npu_root = repo_root / "torchtitan_npu"
    models_root = npu_root / "models"
    vlm_root = models_root / "vlm"

    _ensure_package_stub("torchtitan_npu", npu_root)
    _ensure_package_stub("torchtitan_npu.models", models_root)
    _ensure_package_stub("torchtitan_npu.models.vlm", vlm_root)
    return _load_module(
        "torchtitan_npu.models.vlm.model",
        vlm_root / "model.py",
    )


def _load_vlm_siglip2_module():
    repo_root = Path(__file__).resolve().parents[3]
    npu_root = repo_root / "torchtitan_npu"
    models_root = npu_root / "models"
    vlm_root = models_root / "vlm"

    _ensure_package_stub("torchtitan_npu", npu_root)
    _ensure_package_stub("torchtitan_npu.models", models_root)
    _ensure_package_stub("torchtitan_npu.models.vlm", vlm_root)
    return _load_module(
        "torchtitan_npu.models.vlm.siglip2",
        vlm_root / "siglip2.py",
    )


def _load_vlm_parallelize_module():
    repo_root = Path(__file__).resolve().parents[3]
    npu_root = repo_root / "torchtitan_npu"
    models_root = npu_root / "models"
    vlm_root = models_root / "vlm"

    _ensure_package_stub("torchtitan_npu", npu_root)
    _ensure_package_stub("torchtitan_npu.models", models_root)
    _ensure_package_stub("torchtitan_npu.models.vlm", vlm_root)
    return _load_module(
        "torchtitan_npu.models.vlm.parallelize",
        vlm_root / "parallelize.py",
    )


_, _, _, _multimodal = _load_multimodal_modules()

DenseMaskSDPA = _multimodal.DenseMaskSDPA
build_document_ids = _multimodal.build_document_ids
build_encoder_causal_mask = _multimodal.build_encoder_causal_mask
build_encoder_full_mask = _multimodal.build_encoder_full_mask
build_text_document_causal_mask = _multimodal.build_text_document_causal_mask
build_valid_patch_mask = _multimodal.build_valid_patch_mask
scatter_visual_embeddings = _multimodal.scatter_visual_embeddings


def test_text_document_causal_mask_resets_after_eos():
    tokens = torch.tensor([[10, 11, 2, 20, 21], [30, 2, 31, 32, 2]])

    assert torch.equal(
        build_document_ids(tokens, eos_id=2),
        torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]),
    )

    mask = build_text_document_causal_mask(tokens, eos_id=2)

    expected_first = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, False, False, True, False],
            [False, False, False, True, True],
        ]
    )
    expected_second = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [False, False, True, False, False],
            [False, False, True, True, False],
            [False, False, True, True, True],
        ]
    )
    assert torch.equal(mask[0], expected_first)
    assert torch.equal(mask[1], expected_second)


def test_encoder_causal_mask_keeps_padding_rows_finite():
    valid_tokens = torch.tensor([[True, True, False]])

    mask = build_encoder_causal_mask(valid_tokens)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [
                    [True, False, False],
                    [True, True, False],
                    [False, False, True],
                ]
            ]
        ),
    )

    strict_mask = build_encoder_causal_mask(
        valid_tokens,
        allow_padding_self_attention=False,
    )
    assert torch.equal(
        strict_mask,
        torch.tensor(
            [
                [
                    [True, False, False],
                    [True, True, False],
                    [False, False, False],
                ]
            ]
        ),
    )


def test_encoder_full_mask_allows_bidirectional_valid_patch_attention():
    valid_tokens = torch.tensor([[True, True, False]])

    mask = build_encoder_full_mask(valid_tokens)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, True],
                ]
            ]
        ),
    )

    strict_mask = build_encoder_full_mask(
        valid_tokens,
        allow_padding_self_attention=False,
    )
    assert torch.equal(
        strict_mask,
        torch.tensor(
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, False],
                ]
            ]
        ),
    )


def test_valid_patch_mask_uses_padded_grid_shape_marker():
    grid_hw = torch.tensor([[[2, 3], [-1, 4], [5, -1]]])

    assert torch.equal(
        build_valid_patch_mask(grid_hw),
        torch.tensor([[True, False, False]]),
    )


def test_scatter_visual_embeddings_replaces_image_token_slots():
    hidden_states = torch.zeros(1, 4, 3, dtype=torch.float32)
    tokens = torch.tensor([[7, 42, 8, 42]])
    visual_embeddings = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [99.0, 99.0, 99.0],
                [4.0, 5.0, 6.0],
            ]
        ],
        dtype=torch.float64,
    )
    visual_token_mask = torch.tensor([[True, False, True]])

    output = scatter_visual_embeddings(
        hidden_states,
        tokens,
        visual_embeddings,
        visual_token_mask,
        image_token_id=42,
    )

    assert torch.equal(
        output,
        torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [0.0, 0.0, 0.0],
                    [4.0, 5.0, 6.0],
                ]
            ],
            dtype=torch.float32,
        ),
    )


def test_scatter_visual_embeddings_rejects_count_mismatch():
    hidden_states = torch.zeros(1, 3, 2)
    tokens = torch.tensor([[42, 1, 42]])
    visual_embeddings = torch.ones(1, 1, 2)
    visual_token_mask = torch.tensor([[True]])

    with pytest.raises(ValueError, match="Different number of visual embeddings"):
        scatter_visual_embeddings(
            hidden_states,
            tokens,
            visual_embeddings,
            visual_token_mask,
            image_token_id=42,
        )


def test_scatter_visual_embeddings_rejects_non_bool_visual_mask():
    hidden_states = torch.zeros(1, 1, 2)
    tokens = torch.tensor([[42]])
    visual_embeddings = torch.ones(1, 1, 2)
    visual_token_mask = torch.ones(1, 1)

    with pytest.raises(ValueError, match="visual_token_mask must have dtype torch.bool"):
        scatter_visual_embeddings(
            hidden_states,
            tokens,
            visual_embeddings,
            visual_token_mask,
            image_token_id=42,
        )


def test_scatter_visual_embeddings_reports_actual_shape_mismatch():
    hidden_states = torch.zeros(1, 2, 3)
    tokens = torch.tensor([[42, 1]])
    visual_embeddings = torch.ones(1, 1, 2)
    visual_token_mask = torch.tensor([[True]])

    with pytest.raises(ValueError) as exc_info:
        scatter_visual_embeddings(
            hidden_states,
            tokens,
            visual_embeddings,
            visual_token_mask,
            image_token_id=42,
        )

    message = str(exc_info.value)
    assert "visual_embeddings hidden dim" in message
    assert "(1, 1, 2)" in message
    assert "(1, 2, 3)" in message


def test_scatter_visual_embeddings_can_skip_debug_count_check():
    hidden_states = torch.zeros(1, 1, 2)
    tokens = torch.tensor([[42]])
    visual_embeddings = torch.ones(1, 2, 2)
    visual_token_mask = torch.tensor([[True, True]])

    output = scatter_visual_embeddings(
        hidden_states,
        tokens,
        visual_embeddings,
        visual_token_mask,
        image_token_id=42,
        validate_token_count=False,
    )

    assert torch.equal(output, torch.ones(1, 1, 2))


def test_dense_mask_sdpa_matches_torch_attention_layout():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 2, 3)
    k = torch.randn(2, 4, 2, 3)
    v = torch.randn(2, 4, 2, 3)
    attention_masks = torch.tril(torch.ones(2, 4, 4, dtype=torch.bool))

    output = DenseMaskSDPA()(q, k, v, attention_masks=attention_masks)

    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attention_masks[:, None, :, :],
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)


def test_dense_mask_sdpa_config_triggers_trainer_attention_mask_building():
    from torchtitan.models.common import FlexAttention

    assert isinstance(DenseMaskSDPA.Config(), FlexAttention.Config)


def test_resize_positional_embeddings_skips_nonpositive_spatial_shapes():
    siglip2 = _load_vlm_siglip2_module()
    pos_embeddings = torch.randn(2, 2, 3)
    spatial_shapes = torch.tensor([[0, 5], [2, 2]])

    resized = siglip2.resize_positional_embeddings_zero_padded(pos_embeddings, spatial_shapes, max_length=5)

    assert torch.equal(resized[0], torch.zeros_like(resized[0]))
    assert not torch.equal(resized[1, :4], torch.zeros_like(resized[1, :4]))


def test_vision_attention_npu_does_not_build_upstream_flex_attention(monkeypatch):
    from torchtitan.experiments.vlm.model import siglip2 as upstream_siglip2
    from torchtitan.models.common.attention import LocalMapInnerAttention
    from torchtitan.models.common.linear import Linear

    siglip2 = _load_vlm_siglip2_module()

    def fail_flex_build(self, **kwargs):
        raise AssertionError("unexpected upstream FlexAttention build")

    monkeypatch.setattr(upstream_siglip2.FlexAttention.Config, "build", fail_flex_build)

    class CountingInnerAttention(LocalMapInnerAttention):
        build_count = 0

        @dataclass(kw_only=True, slots=True)
        class Config(LocalMapInnerAttention.Config):
            pass

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            type(self).build_count += 1

        def forward(self, q, k, v, **kwargs):
            return q

    config = siglip2.VisionAttentionNpu.Config(
        qkv_proj=Linear.Config(in_features=4, out_features=4),
        out_proj=Linear.Config(in_features=4, out_features=4),
        n_heads=2,
        dim=4,
        inner_attention=CountingInnerAttention.Config(),
    )

    attention = siglip2.VisionAttentionNpu(config)

    assert isinstance(attention.inner_attention, CountingInnerAttention)
    assert CountingInnerAttention.build_count == 1


def test_vision_attention_npu_tracks_upstream_attention_initialized_state():
    from torchtitan.experiments.vlm.model import siglip2 as upstream_siglip2
    from torchtitan.models.common.linear import Linear

    siglip2 = _load_vlm_siglip2_module()
    upstream_config = upstream_siglip2.Attention.Config(
        qkv_proj=Linear.Config(in_features=4, out_features=4),
        out_proj=Linear.Config(in_features=4, out_features=4),
        n_heads=2,
        dim=4,
    )
    npu_config = siglip2.VisionAttentionNpu.Config(
        qkv_proj=upstream_config.qkv_proj,
        out_proj=upstream_config.out_proj,
        n_heads=upstream_config.n_heads,
        dim=upstream_config.dim,
        inner_attention=DenseMaskSDPA.Config(),
    )

    upstream_attention = upstream_siglip2.Attention(upstream_config)
    npu_attention = siglip2.VisionAttentionNpu(npu_config)

    upstream_keys = set(upstream_attention.__dict__)
    npu_keys = set(npu_attention.__dict__)
    assert upstream_keys == npu_keys, (
        "NPU VisionAttentionNpu instance fields differ from upstream "
        f"Attention. Missing: {upstream_keys - npu_keys}; "
        f"extra: {npu_keys - upstream_keys}."
    )

    upstream_modules = {name for name, _ in upstream_attention.named_children()}
    npu_modules = {name for name, _ in npu_attention.named_children()}
    assert upstream_modules == npu_modules, (
        "NPU VisionAttentionNpu modules differ from upstream Attention. "
        f"Missing: {upstream_modules - npu_modules}; "
        f"extra: {npu_modules - upstream_modules}."
    )


def test_vision_transformer_layer_npu_does_not_build_upstream_attention(monkeypatch):
    from torchtitan.experiments.vlm.model import siglip2 as upstream_siglip2
    from torchtitan.models.common.linear import Linear

    siglip2 = _load_vlm_siglip2_module()

    def fail_upstream_attention_init(self, config):
        raise AssertionError("unexpected upstream Attention construction")

    monkeypatch.setattr(
        upstream_siglip2.Attention,
        "__init__",
        fail_upstream_attention_init,
    )

    config = siglip2.VisionTransformerLayerNpu.Config(
        self_attn=siglip2.VisionAttentionNpu.Config(
            qkv_proj=Linear.Config(in_features=4, out_features=4),
            out_proj=Linear.Config(in_features=4, out_features=4),
            n_heads=2,
            dim=4,
            inner_attention=DenseMaskSDPA.Config(),
        ),
        mlp=upstream_siglip2.FeedForward.Config(
            fc1=Linear.Config(in_features=4, out_features=8, bias=True),
            fc2=Linear.Config(in_features=8, out_features=4, bias=True),
        ),
        layer_norm_eps=1e-6,
        dim=4,
    )

    layer = siglip2.VisionTransformerLayerNpu(config)

    assert isinstance(layer.self_attn, siglip2.VisionAttentionNpu)


def test_vision_transformer_npu_does_not_build_upstream_layers(monkeypatch):
    from torchtitan.models.common.embedding import Embedding
    from torchtitan.models.common.linear import Linear

    siglip2 = _load_vlm_siglip2_module()

    def fail_upstream_layer_init(self, config):
        raise AssertionError("unexpected upstream TransformerLayer construction")

    monkeypatch.setattr(
        siglip2.upstream_siglip2.TransformerLayer,
        "__init__",
        fail_upstream_layer_init,
    )

    config = siglip2.VisionTransformerNpu.Config(
        dim=4,
        embeddings=siglip2.VisionEmbeddingsNpu.Config(
            patch_embedding=Linear.Config(in_features=4, out_features=4),
            position_embedding=Embedding.Config(num_embeddings=4, embedding_dim=4),
            n_pos_embs=2,
        ),
        layers=[
            siglip2.VisionTransformerLayerNpu.Config(
                self_attn=siglip2.VisionAttentionNpu.Config(
                    qkv_proj=Linear.Config(in_features=4, out_features=4),
                    out_proj=Linear.Config(in_features=4, out_features=4),
                    n_heads=2,
                    dim=4,
                    inner_attention=DenseMaskSDPA.Config(),
                ),
                mlp=siglip2.upstream_siglip2.FeedForward.Config(
                    fc1=Linear.Config(in_features=4, out_features=8, bias=True),
                    fc2=Linear.Config(in_features=8, out_features=4, bias=True),
                ),
                layer_norm_eps=1e-6,
                dim=4,
            )
        ],
        n_channels=3,
        patch_size=16,
        layer_norm_eps=1e-6,
        attn_mask_type="causal",
    )

    transformer = siglip2.VisionTransformerNpu(config)

    assert isinstance(transformer.embeddings, siglip2.VisionEmbeddingsNpu)
    assert isinstance(transformer.layers["0"], siglip2.VisionTransformerLayerNpu)


def test_vlm_config_uses_model_scoped_tokenizer_assets():
    config_registry = _load_vlm_config_registry()
    config = config_registry.vlm_debugmodel_npu()

    assert config.hf_assets_path == "./tests/assets/tokenizer/vlm_tokenizer"


def test_vlm_config_uses_explicit_npu_vlm_converter():
    from torchtitan_npu.converters.registry import has_npu_converter

    config_registry = _load_vlm_config_registry()
    config = config_registry.vlm_debugmodel_npu()

    assert has_npu_converter(config.model_converters.converters, "npu_vlm")


def test_vlm_config_uses_npu_profiling_config():
    from torchtitan_npu.config.configs import ProfilingConfig

    config_registry = _load_vlm_config_registry()
    config = config_registry.vlm_debugmodel_npu()

    assert isinstance(config.profiling, ProfilingConfig)
    assert config.profiling.enable_profiling is False


def test_vlm_compile_config_does_not_mutate_base_config(monkeypatch):
    config_registry = _load_vlm_config_registry()
    base_config = config_registry.vlm_debugmodel_npu()

    monkeypatch.setattr(config_registry, "vlm_debugmodel_npu", lambda: base_config)

    compile_config = config_registry.vlm_debugmodel_npu_compile()

    assert compile_config is not base_config
    assert compile_config.compile.enable
    assert not base_config.compile.enable


def test_vlm_model_registry_returns_npu_model_config():
    vlm_module = _load_module(
        "torchtitan_npu.models.vlm",
        Path(__file__).resolve().parents[3] / "torchtitan_npu" / "models" / "vlm" / "__init__.py",
    )
    vlm_model = importlib.import_module("torchtitan_npu.models.vlm.model")

    spec = vlm_module.model_registry("debugmodel")

    assert isinstance(spec.model, vlm_model.Llama3Siglip2TransformerNpu.Config)


def test_vlm_model_registry_does_not_mutate_upstream_symbols():
    upstream_model = importlib.import_module("torchtitan.experiments.vlm.model.model")
    upstream_siglip2 = importlib.import_module("torchtitan.experiments.vlm.model.siglip2")
    original_get_attention_masks = upstream_model.Llama3Siglip2Transformer.get_attention_masks
    original_scatter = _module_symbol(upstream_model, "_scatter_img_tokens")
    original_resize = upstream_siglip2.resize_positional_embeddings
    original_attention_forward = upstream_siglip2.Attention.forward

    vlm_module = _load_module(
        "torchtitan_npu.models.vlm",
        Path(__file__).resolve().parents[3] / "torchtitan_npu" / "models" / "vlm" / "__init__.py",
    )
    vlm_module.model_registry("debugmodel")

    assert upstream_model.Llama3Siglip2Transformer.get_attention_masks is original_get_attention_masks
    assert _module_symbol(upstream_model, "_scatter_img_tokens") is original_scatter
    assert upstream_siglip2.resize_positional_embeddings is original_resize
    assert upstream_siglip2.Attention.forward is original_attention_forward


def test_npu_vlm_config_builds_dense_attention_without_runtime_replacement():
    import torchtitan_npu.converters.features.vlm as vlm_converter
    from torchtitan_npu.models.vlm import model_registry

    spec = model_registry("debugmodel")
    with torch.device("meta"):
        model = spec.model.build()

    text_attention = model.layers["0"].attention.inner_attention
    vision_attention = model.encoder.layers["0"].self_attn.inner_attention

    assert isinstance(text_attention, DenseMaskSDPA)
    assert isinstance(vision_attention, DenseMaskSDPA)

    vlm_converter.NpuVLMConverter(spec).convert(model)

    assert model.layers["0"].attention.inner_attention is text_attention
    assert model.encoder.layers["0"].self_attn.inner_attention is vision_attention


def test_npu_vlm_converter_rejects_non_dense_attention():
    from torchtitan.models.common import FlexAttention

    import torchtitan_npu.converters.features.vlm as vlm_converter
    from torchtitan_npu.models.vlm import model_registry

    spec = model_registry("debugmodel")
    with torch.device("meta"):
        model = spec.model.build()

    model.layers["0"].attention.inner_attention = FlexAttention.Config().build()

    with pytest.raises(TypeError, match="decoder DenseMaskSDPA"):
        vlm_converter.NpuVLMConverter(spec).convert(model)


def test_npu_vlm_attention_masks_include_pixel_masks():
    vlm_model = _load_vlm_model_module()
    model = vlm_model.Llama3Siglip2TransformerNpu.__new__(vlm_model.Llama3Siglip2TransformerNpu)
    grid_thw = torch.tensor([[[0, 0, 0], [0, 0, 1], [-1, -1, -1]]])

    masks = model.get_attention_masks(
        torch.tensor([[10, 2, 11]]),
        SimpleNamespace(eos_id=2),
        {"grid_thw": grid_thw},
    )

    assert torch.equal(
        masks["pixel_masks"],
        build_valid_patch_mask(grid_thw[:, :, 1:]),
    )


def test_npu_vlm_forward_reuses_pixel_masks_from_attention_masks(monkeypatch):
    vlm_model = _load_vlm_model_module()
    model = vlm_model.Llama3Siglip2TransformerNpu.__new__(vlm_model.Llama3Siglip2TransformerNpu)
    nn.Module.__init__(model)

    pixel_masks = torch.tensor([[True, False]])
    encoder_masks = torch.ones(1, 2, 2, dtype=torch.bool)
    llama3_masks = torch.ones(1, 3, 3, dtype=torch.bool)
    tokens = torch.zeros(1, 3, 4)
    pixel_values = torch.zeros(1, 2, 4)
    grid_thw = torch.tensor([[[0, 0, 0], [-1, -1, -1]]])

    class FakeEncoder:
        @staticmethod
        def __call__(values, masks, grid_hw, attention_masks):
            assert masks is pixel_masks
            assert attention_masks is encoder_masks
            return values

    def fail_build_valid_patch_mask(grid_hw):
        raise AssertionError("forward should reuse pixel_masks from attention_masks")

    def fake_scatter(hidden_states, token_ids, visual_embeddings, masks, image_token_id):
        assert masks is pixel_masks
        return hidden_states

    monkeypatch.setattr(vlm_model, "build_valid_patch_mask", fail_build_valid_patch_mask)
    monkeypatch.setattr(vlm_model, "scatter_visual_embeddings", fake_scatter)

    model.tok_embeddings = None
    model.encoder = FakeEncoder()
    model.projector = nn.Identity()
    model.layers = {}
    model.norm = None
    model.output = None

    output = model.forward(
        tokens,
        pixel_values,
        grid_thw,
        SimpleNamespace(img_id=42),
        attention_masks={
            "encoder_masks": encoder_masks,
            "llama3_masks": llama3_masks,
            "pixel_masks": pixel_masks,
        },
    )

    assert output is tokens


def test_cc12m_test_dataset_matches_upstream_asset_path():
    from torchtitan.experiments.vlm.datasets.mm_datasets import MM_DATASETS
    from torchtitan.models.flux.flux_datasets import DATASETS as FLUX_DATASETS

    expected_path = "tests/assets/cc12m_test"

    assert MM_DATASETS["cc12m-test"].path == expected_path
    assert FLUX_DATASETS["cc12m-test"].path == expected_path


def test_vlm_config_uses_cc12m_test_registry_default_path():
    config_registry = _load_vlm_config_registry()
    config = config_registry.vlm_debugmodel_npu()

    assert config.dataloader.dataset == "cc12m-test"
    assert config.dataloader.dataset_path is None


def _parallelize_vlm_npu_for_test(parallelize, parallel_dims, model_converters=None):
    return parallelize.parallelize_vlm_npu(
        MagicMock(),
        parallel_dims=parallel_dims,
        training=MagicMock(),
        model_converters=model_converters or ModelConvertersContainer.Config(),
        parallelism=MagicMock(),
        compile_config=MagicMock(),
        ac_config=MagicMock(),
        dump_folder="/tmp/test",
    )


def test_parallelize_vlm_npu_rejects_non_fsdp_parallelism():
    parallelize = _load_vlm_parallelize_module()
    parallel_dims = SimpleNamespace(
        tp_enabled=True,
        pp_enabled=True,
        cp_enabled=True,
    )

    with pytest.raises(NotImplementedError, match="only supports FSDP/HSDP data parallelism"):
        _parallelize_vlm_npu_for_test(parallelize, parallel_dims)


def test_parallelize_vlm_npu_validates_parallel_dims_before_converter_check():
    parallelize = _load_vlm_parallelize_module()
    parallel_dims = SimpleNamespace(
        tp_enabled=True,
        pp_enabled=False,
        cp_enabled=False,
    )

    with pytest.raises(NotImplementedError, match="only supports FSDP/HSDP"):
        _parallelize_vlm_npu_for_test(parallelize, parallel_dims)


def test_parallelize_vlm_npu_requires_npu_vlm_converter():
    parallelize = _load_vlm_parallelize_module()
    parallel_dims = SimpleNamespace(
        tp_enabled=False,
        pp_enabled=False,
        cp_enabled=False,
    )

    with pytest.raises(ValueError, match='requires "npu_vlm"'):
        _parallelize_vlm_npu_for_test(parallelize, parallel_dims)
