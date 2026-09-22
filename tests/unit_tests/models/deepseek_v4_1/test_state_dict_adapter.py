# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 StateDict adapter round-trip test: verify ownership-based partition."""

import torch

from torchtitan_npu.models.deepseek_v4_1 import (
    model_registry,
)
from torchtitan_npu.models.deepseek_v4_1.state_dict_adapter import DeepSeekV41StateDictAdapter
from torchtitan_npu.models.deepseek_v4_1.vision.state_dict_adapter import DeepSeekV41VisionStateDictAdapter


def _build_model_config():
    """A minimal local V4.1 fixture (the debugmodel topology, tiny widths)."""
    config = model_registry("deepseek_v4_1_debugmodel").model
    for layer in config.layers:
        layer.engram = None
    config.vocab_size = 32
    config.tok_embeddings.num_embeddings = 32
    config.lm_head.out_features = 32
    return config


def _vision_hf_dict() -> dict[str, torch.Tensor]:
    return {
        # Vision tower
        "vision.patch_embed.proj.weight": torch.randn(32, 3, 14, 14),
        "vision.norm.weight": torch.randn(32),
        "aligner.w1.weight": torch.randn(64, 32),
        "image_start": torch.randn(32),
        "image_newline": torch.randn(32),
        "image_end": torch.randn(32),
        # The router's vision bias: a language-MoE parameter that only a modality stack
        # builds, so the vision adapter owns its mapping.
        "layers.3.ffn.gate.bias_vl": torch.randn(4),
    }


def _base_hf_dict() -> dict[str, torch.Tensor]:
    return {
        # Text backbone (keys match the V4.1 adapter's from_hf_map).
        "embed.weight": torch.randn(512, 32),
        "layers.3.attn.wq_a.weight": torch.randn(8, 32),
        "layers.3.attn.wq_b.weight": torch.randn(16, 8),
        "layers.3.ffn.gate.weight": torch.randn(4, 32),
        "layers.3.ffn.gate.bias": torch.randn(4),
        "norm.weight": torch.randn(32),
    }


def _merged_hf_dict() -> dict[str, torch.Tensor]:
    return _base_hf_dict() | _vision_hf_dict()


class TestVisionAdapterOwnership:
    """Unit tests for DeepSeekV41VisionStateDictAdapter ownership helpers."""

    @staticmethod
    def test_owns_hf_key():
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("vision.patch_embed.proj.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("aligner.w1.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_start")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_newline")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("image_end")
        assert DeepSeekV41VisionStateDictAdapter.owns_hf_key("layers.3.ffn.gate.bias_vl")
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("embed.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("layers.0.attn.wq_a.weight")
        # The rest of the layer namespace stays the text half's.
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("layers.0.ffn.gate.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_hf_key("layers.0.ffn.gate.bias")

    @staticmethod
    def test_owns_local_key():
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("vision_encoder.patch_embed.proj.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("vision_encoder.aligner.w1.weight")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("image_marker_embeddings.image_start")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("image_marker_embeddings.image_end")
        assert DeepSeekV41VisionStateDictAdapter.owns_local_key("layers.3.moe.router.bias_vl")
        assert not DeepSeekV41VisionStateDictAdapter.owns_local_key("embed.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_local_key("layers.0.attention.wq_a.weight")
        assert not DeepSeekV41VisionStateDictAdapter.owns_local_key("layers.0.moe.router.gate.weight")

    @staticmethod
    def test_vision_adapter_round_trip():
        """Vision-only: HF→local→HF preserves key set and name format."""
        adapter = DeepSeekV41VisionStateDictAdapter()
        hf_in = _vision_hf_dict()
        local = adapter.from_hf(hf_in)
        hf_out = adapter.to_hf(local)

        # Every HF key has a local counterpart (namespaced properly).
        for hf_key in hf_in:
            assert hf_key in hf_out, f"missing {hf_key} in round-trip"
        # No raw HF-style key leaked into local.
        for lk in local:
            assert (
                lk.startswith("vision_encoder.")
                or lk.startswith("image_marker_embeddings.")
                or lk.endswith(".moe.router.bias_vl")
            ), f"local key {lk} is not in vision namespace"


class TestComposedAdapterPartition:
    """Ownership partition ensures no key-space pollution."""

    @staticmethod
    def test_partition_prevents_duplicate_keys():
        adapter = DeepSeekV41StateDictAdapter(
            model_config=_build_model_config(),
            hf_assets_path=None,
        )
        hf_in = _merged_hf_dict()
        local = adapter.from_hf(hf_in)

        # Base keys: raw HF name must NOT appear in local (the adapter
        # always transforms them).
        assert "embed.weight" not in local, "raw HF embed.weight leaked"
        assert "layers.0.attn.wq_a.weight" not in local, "raw HF attn key leaked"
        # Vision keys: raw HF name must NOT appear in local.
        assert "vision.patch_embed.proj.weight" not in local, "raw HF vision key leaked"

        # Every key in local has the expected prefix.
        expected_prefixes = ("tok_embeddings", "layers.", "vision_encoder.", "image_marker_embeddings.", "norm.")
        for lk in local:
            assert any(lk.startswith(p) for p in expected_prefixes), f"unexpected local key {lk}"

        # Round-trip: local → HF must reproduce the original HF key set
        # with bit-identical values (strict contract, no tautology).
        hf_out = adapter.to_hf(local)
        assert set(hf_out) == set(hf_in), (
            f"HF round-trip key set mismatch: missing={set(hf_in) - set(hf_out)}, extra={set(hf_out) - set(hf_in)}"
        )
        for key in hf_in:
            assert torch.equal(hf_out[key], hf_in[key]), f"HF round-trip value mismatch for {key}"

    @staticmethod
    def test_round_trip_deterministic():
        adapter = DeepSeekV41StateDictAdapter(
            model_config=_build_model_config(),
            hf_assets_path=None,
        )
        hf_in = _merged_hf_dict()
        local1 = adapter.from_hf(hf_in)
        local2 = adapter.from_hf(hf_in)
        assert set(local1.keys()) == set(local2.keys())
        for k in local1:
            assert torch.equal(local1[k], local2[k]), f"key {k} differs"


class TestAdapterPartitionIsTotal:
    """The two adapters must partition the key space, not overlap or leave gaps.

    ``owns_hf_key`` is the dispatch predicate: the composed adapter splits on it and
    never re-checks.  So a key claimed by neither adapter is silently unmapped and a key
    claimed by both is silently routed to whichever half runs first.
    """

    # The key spellings the released V4.1 checkpoint uses, by namespace.
    _TEXT_KEYS = (
        "embed.weight",
        "head.weight",
        "norm.weight",
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.compressor.wkv.weight",
        "layers.0.attn.indexer.wk.weight",
        "layers.0.ffn.gate.weight",
        "layers.0.ffn.gate.bias",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.0.hc_attn_fn",
    )
    _VISION_KEYS = (
        "vision.patch_embed.proj.weight",
        "vision.blocks.0.attn.wqkv.weight",
        "vision.norm.weight",
        "aligner.w1.weight",
        "image_start",
        "image_newline",
        "image_end",
        "layers.0.ffn.gate.bias_vl",
    )

    def test_every_key_is_claimed_exactly_once(self):
        vision_owns = DeepSeekV41VisionStateDictAdapter.owns_hf_key
        for key in self._TEXT_KEYS:
            assert not vision_owns(key), f"the vision adapter must not claim {key}"
        for key in self._VISION_KEYS:
            assert vision_owns(key), f"the vision adapter must claim {key}"

    def test_bias_vl_is_the_only_layer_scoped_vision_key(self):
        """``bias_vl`` sits in the text half's namespace; nothing else there does."""
        vision_owns = DeepSeekV41VisionStateDictAdapter.owns_hf_key
        for layer in range(4):
            for key in (f"layers.{layer}.ffn.gate.weight", f"layers.{layer}.ffn.gate.bias"):
                assert not vision_owns(key), key
            assert vision_owns(f"layers.{layer}.ffn.gate.bias_vl")
        # A near-miss spelling must not be claimed by pattern.
        assert not vision_owns("layers.0.ffn.gate.bias_vl_extra")
        assert not vision_owns("layers.0.ffn.gate.bias_vl.extra")
        assert not vision_owns("mtp.0.ffn.gate.bias_vl")


class TestTextFlavorAdapter:
    """A text-only model has no vision half, so it gets no vision rule at all."""

    @staticmethod
    def _text_adapter():
        config = model_registry("deepseek_v4_1_debugmodel_text").model
        for layer in config.layers:
            layer.engram = None
        config.vocab_size = 32
        config.tok_embeddings.num_embeddings = 32
        config.lm_head.out_features = 32
        return DeepSeekV41StateDictAdapter(model_config=config, hf_assets_path=None)

    def test_no_vision_adapter_and_no_bias_vl_rule(self):
        adapter = self._text_adapter()
        assert adapter._vision_adapter is None
        # The rule the vision half owns must not survive in the text half's map: with no
        # parameter to land on, mapping it would produce an orphan checkpoint key.
        assert "layers.{}.ffn.gate.bias_vl" not in adapter.from_hf_map
        assert not any(adapter._owns_vision_hf_key(k) for k in TestAdapterPartitionIsTotal._VISION_KEYS)

    def test_text_half_round_trips_without_vision_keys(self):
        adapter = self._text_adapter()
        hf_in = {
            "embed.weight": torch.randn(32, 32),
            "layers.3.attn.wq_a.weight": torch.randn(8, 32),
            "layers.3.ffn.gate.weight": torch.randn(4, 32),
            "layers.3.ffn.gate.bias": torch.randn(4),
            "norm.weight": torch.randn(32),
        }
        local = adapter.from_hf(hf_in)
        assert "layers.3.moe.router.bias_vl" not in local
        hf_out = adapter.to_hf(local)
        assert set(hf_out) == set(hf_in)
