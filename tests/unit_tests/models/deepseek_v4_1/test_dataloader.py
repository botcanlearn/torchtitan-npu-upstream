# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV4.1 protocol, upstream DP sharding, packing and checkpoint contracts."""

from copy import deepcopy
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.deepseek_v4_1.dataloader import DeepSeekV41DataLoader
from torchtitan_npu.models.deepseek_v4_1.model import V41Model
from torchtitan_npu.models.deepseek_v4_1.vision_data import IMAGE_END, IMAGE_START, ImagePatchProcessor

ASSETS = Path(__file__).resolve().parents[3] / "assets"


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceTokenizer(tokenizer_path=str(ASSETS / "deepseek_v3"))


def _loader(tokenizer, *, rank=0, world=1, packing=0, infinite=False, alignment=2):
    return DeepSeekV41DataLoader.Config(
        dataset="cc12m-test",
        dataset_path=str(ASSETS / "cc12m_test"),
        packing_buffer_size=packing,
        infinite=infinite,
        document_alignment=alignment,
    ).build(tokenizer=tokenizer, dp_rank=rank, dp_world_size=world, seq_len=512, local_batch_size=1)


def _captions(loader):
    return [tuple(labels[labels != -100].tolist()) for _, labels in loader]


def test_upstream_webdataset_dp_shards(tokenizer):
    expected = _captions(_loader(tokenizer))
    shards = [_captions(_loader(tokenizer, rank=rank, world=2)) for rank in range(2)]
    assert len(expected) == len(set(expected)) == 32
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == sorted(expected)


def test_documents_align_to_the_pooling_ratio_and_marks_only_padding(tokenizer):
    """Every document ends on a multiple of the alignment; validity follows the
    real tokens, so the compressed groups never straddle a document edge."""
    saw_odd_document = False
    for inputs, _ in _loader(tokenizer):
        tokens = inputs["input"][0]
        positions = inputs["positions"][0]
        valid = inputs["valid_tokens"][0]
        starts = (positions == 0).nonzero().flatten().tolist()
        # Every document — the real one and the row-tail padding document —
        # has an even token count, including its structural padding.
        for begin, end in pairwise([*starts, tokens.numel()]):
            assert (end - begin) % 2 == 0, (begin, end)
        real_length = int(valid.sum())
        if real_length % 2:
            saw_odd_document = True
        # True exactly on the real tokens: the alignment pad and the row tail
        # are the only False span, and it starts right after the real EOS.
        assert torch.equal(valid[:real_length], torch.ones(real_length, dtype=torch.bool))
        assert not valid[real_length:].any()
        assert tokens[real_length - 1] == tokenizer.eos_id
    # The committed tar really contains odd-length documents to align.
    assert saw_odd_document


def test_packed_images_supervision_and_positions(tokenizer):
    unpacked = _captions(_loader(tokenizer))
    loader = _loader(tokenizer, packing=4)
    iterator = iter(loader)
    first = next(iterator)
    assert loader.dataset._sample_idx == 4  # produce before EOF without an exact-length fit
    packed = [first, *iterator]
    assert len(packed) < len(unpacked)
    actual_captions = []
    for inputs, labels in packed:
        tokens, positions, types, indices, valid = [
            inputs[k][0] for k in ("input", "positions", "token_types", "image_feature_indices", "valid_tokens")
        ]
        starts = (tokens == tokenizer.bos_id).nonzero().flatten().tolist()
        ends = (tokens == tokenizer.eos_id).nonzero().flatten().tolist()
        assert len(starts) == len(ends) == inputs["pixel_values"].shape[0]
        assert (types == IMAGE_START).sum() == (types == IMAGE_END).sum() == len(starts)
        assert torch.equal(indices[indices >= 0], torch.arange((indices >= 0).sum()))
        # Every packed document — including the tail padding document — is
        # aligned, and the validity mark is True exactly on the real spans.
        for begin, end in pairwise([*starts, tokens.numel()]):
            assert (end - begin) % 2 == 0, (begin, end)
        expected_valid = torch.zeros_like(valid)
        for start, end in zip(starts, ends, strict=True):
            assert torch.equal(positions[start : end + 1], torch.arange(end - start + 1))
            expected_valid[start : end + 1] = True
            actual_captions.append(tuple(labels[0, start:end][labels[0, start:end] != -100].tolist()))
            assert labels[0, end] == -100  # no cross-document BOS prediction
        assert torch.equal(valid, expected_valid)
        assert (labels[0, :-1][types[1:] >= 0] == -100).all()
        mask = labels[0, :-1] != -100
        assert torch.equal(labels[0, :-1][mask], tokens[1:][mask])
        assert (labels[0, int(ends[-1]) :] == -100).all()
    assert sorted(actual_captions) == sorted(unpacked)


@pytest.mark.parametrize("consumed", [1, 33])
def test_packed_loader_resume_pending_samples(tokenizer, consumed):
    original = _loader(tokenizer, packing=8, infinite=True)
    iterator = iter(original)
    for _ in range(consumed):
        next(iterator)
    state = deepcopy(original.state_dict())
    restored = _loader(tokenizer, packing=8, infinite=True)
    restored.load_state_dict(state)
    resumed = iter(restored)
    # Restore buffered packs both in the first epoch and after all 32 source samples.
    for _ in range(5):
        expected, expected_labels = next(iterator)
        actual, actual_labels = next(resumed)
        assert torch.equal(actual_labels, expected_labels)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert torch.equal(actual[key], expected[key]), key


def test_target_grid_fixed_oracle():
    processor = ImagePatchProcessor()
    expected = {(100, 100): (39, 39), (60, 1600): (201, 8), (1600, 60): (8, 201),
                (701, 1024): (74, 51), (555, 416): (34, 45)}
    for (width, height), grid in expected.items():
        assert processor.target_grid(height, width) == grid


def test_recipe_passes_the_model_alignment_to_the_loader():
    """The recipes derive the loader's document_alignment from the model
    spec (its max pooling ratio), so a packed document never straddles a
    pooled group."""
    from torchtitan_npu.models.deepseek_v4_1 import config_registry as cr

    trainer = cr.deepseek_v4_1_debugmodel_multimodal()
    assert trainer.dataloader.document_alignment == cr._document_alignment(trainer.model_spec) == 2


def test_valid_tokens_reach_the_attention_metadata(tokenizer):
    """The collator's validity mark is consumed by the model-owned metadata
    hook (not forwarded to the model) and lands in the metadata the
    attention reads; the ragged boundaries come from the same positions."""
    loader = _loader(tokenizer, packing=4)
    inputs, labels = next(iter(loader))
    extra = {key: value.clone() for key, value in inputs.items()}

    class _HookOwner:
        # call the real hook through a minimal owner: the hook dispatches to
        # the model's own metadata builder, so attach the actual construction
        # methods and the ratio table its selection-mask precompute reads
        compress_ratios = (2,)
        build_attention_masks = V41Model.build_attention_masks
        get_attention_masks = V41Model.get_attention_masks

    _, _, extra = _HookOwner().build_attention_masks(inputs, labels, extra)

    assert "valid_tokens" not in extra  # consumed by the hook, not forwarded
    metadata = extra["attention_masks"]
    torch.testing.assert_close(metadata.valid_tokens_BL, inputs["valid_tokens"], rtol=0, atol=0)
    starts = (inputs["positions"][0] == 0).nonzero().flatten().to(torch.int32)
    expected_cu = torch.cat((starts, torch.tensor([inputs["positions"].numel()], dtype=torch.int32)))
    torch.testing.assert_close(metadata.cu_seq_q, expected_cu, rtol=0, atol=0)
