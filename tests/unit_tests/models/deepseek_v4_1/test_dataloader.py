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

from torchtitan_npu.models.deepseek_v4_1.model import V41Model
from torchtitan_npu.models.deepseek_v4_1.vision.data import IMAGE_END, IMAGE_START, ImagePatchProcessor
from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import DeepSeekV41DataLoader

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
        per_doc_alignment=alignment,
    ).build(tokenizer=tokenizer, dp_rank=rank, dp_world_size=world, seq_len=512, local_batch_size=1)


def _captions(loader):
    return [tuple(labels[labels != -100].tolist()) for _, labels in loader]


def test_upstream_webdataset_dp_shards(tokenizer):
    expected = _captions(_loader(tokenizer))
    shards = [_captions(_loader(tokenizer, rank=rank, world=2)) for rank in range(2)]
    assert len(expected) == len(set(expected)) == 32
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == sorted(expected)


def test_documents_align_to_the_pooling_ratio(tokenizer):
    """Every document ends on a multiple of the alignment, so a compressor group
    never straddles a document edge."""
    saw_odd_document = False
    for inputs, _ in _loader(tokenizer):
        tokens = inputs["input"][0]
        positions = inputs["positions"][0]
        starts = (positions == 0).nonzero().flatten().tolist()
        # Every document -- the real one and the row-tail padding document --
        # has an even token count, including its alignment pad.
        for begin, end in pairwise([*starts, tokens.numel()]):
            assert (end - begin) % 2 == 0, (begin, end)
        # The pack carries at least one real document, so the even lengths above are
        # padded documents rather than an empty stream.
        assert (tokens == tokenizer.eos_id).any()
    # The committed tar really contains odd-length documents to align: without the
    # pad at least one document would have an odd length.
    for inputs, labels in _loader(tokenizer, alignment=1):
        tokens = inputs["input"][0]
        positions = inputs["positions"][0]
        starts = (positions == 0).nonzero().flatten().tolist()
        if any((end - begin) % 2 for begin, end in pairwise([*starts, tokens.numel()])):
            saw_odd_document = True
    assert saw_odd_document


def test_alignment_pads_documents_without_training_them(tokenizer):
    """A document's supervised content is the same aligned or not: the pad rounds the
    document up and no document is dropped or reordered."""
    saw_odd = False
    odd_documents = []
    for alignment in (1, 2):
        documents = []
        for inputs, labels in _loader(tokenizer, alignment=alignment):
            tokens = inputs["input"][0]
            starts = (inputs["positions"][0] == 0).nonzero().flatten().tolist()
            for begin, end in pairwise([*starts, tokens.numel()]):
                if alignment == 2:
                    assert (end - begin) % 2 == 0, (begin, end)
                row_labels = labels[0, begin:end]
                mask = row_labels != -100
                # A supervised target is the next token of the same row, never a pad.
                assert torch.equal(row_labels[:-1][mask[:-1]], tokens[begin + 1 : end][mask[:-1]])
                supervised = tuple(row_labels[mask].tolist())
                documents.append(supervised)
                if alignment == 1 and (end - begin) % 2:
                    saw_odd = True
                    odd_documents.append(supervised)
        if alignment == 1:
            unaligned = documents
        else:
            aligned = documents
    # The tar really contains odd-length documents, and each is still emitted.
    assert saw_odd
    for document in odd_documents:
        assert document in aligned
    assert len(unaligned) == len(aligned)


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
        tokens, positions, types, indices = [
            inputs[k][0] for k in ("input", "positions", "token_types", "image_feature_indices")
        ]
        starts = (tokens == tokenizer.bos_id).nonzero().flatten().tolist()
        ends = (tokens == tokenizer.eos_id).nonzero().flatten().tolist()
        assert len(starts) == len(ends) == inputs["pixel_values"].shape[0]
        assert (types == IMAGE_START).sum() == (types == IMAGE_END).sum() == len(starts)
        assert torch.equal(indices[indices >= 0], torch.arange((indices >= 0).sum()))
        for start, end in zip(starts, ends, strict=True):
            assert torch.equal(positions[start : end + 1], torch.arange(end - start + 1))
            actual_captions.append(tuple(labels[0, start:end][labels[0, start:end] != -100].tolist()))
            assert labels[0, end] == -100  # no cross-document BOS prediction
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
    expected = {
        (100, 100): (39, 39),
        (60, 1600): (201, 8),
        (1600, 60): (8, 201),
        (701, 1024): (74, 51),
        (555, 416): (34, 45),
    }
    for (width, height), grid in expected.items():
        assert processor.target_grid(height, width) == grid


def test_mm_sample_is_padded_to_the_alignment(tokenizer):
    """The multimodal sample parser rounds each document up to the alignment."""
    import tarfile

    from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import _process_mm_sample

    with tarfile.open(ASSETS / "cc12m_test" / "cc12m-train-0000.tar") as tar:
        members = {name: tar.extractfile(name).read() for name in tar.getnames()}
    stems = sorted(name[:-4] for name in members if name.endswith(".txt"))
    # A sample whose token stream is odd, so the pad actually fires.
    for stem in stems:
        sample = {
            "jpg": members[f"{stem}.jpg"],
            "txt": members[f"{stem}.txt"].decode("utf-8", "replace").strip(),
        }
        unaligned = _process_mm_sample(dict(sample), tokenizer, per_doc_alignment=1)
        if int(unaligned["input_ids"].numel()) % 2:
            break
    else:
        pytest.skip("no odd-length document in the test tar")

    aligned = _process_mm_sample(dict(sample), tokenizer, per_doc_alignment=2)
    # One pad, appended; the real ids are a prefix and the pad is unsupervised.
    assert int(aligned["input_ids"].numel()) == int(unaligned["input_ids"].numel()) + 1
    assert aligned["input_ids"][: unaligned["input_ids"].numel()].tolist() == unaligned["input_ids"].tolist()
    assert (aligned["labels"][unaligned["labels"].numel() :] == -100).all()
    # Positions keep running through the pad, so it stays inside its document.
    assert torch.equal(aligned["positions"], torch.arange(aligned["input_ids"].numel()))


def test_mm_loader_rejects_an_alignment_that_does_not_divide_seq_len(tokenizer):
    with pytest.raises(ValueError, match="per_doc_alignment"):
        _loader(tokenizer, alignment=3)
    with pytest.raises(ValueError, match="per_doc_alignment"):
        _loader(tokenizer, alignment=0)


def test_recipe_passes_the_model_alignment_to_the_loader():
    """The recipes derive the loader's per-document alignment from the model spec
    (the LCM of its compression ratios), so a packed document never straddles a
    pooled group."""
    from torchtitan_npu.models.deepseek_v4_1 import config_registry as cr

    trainer = cr.deepseek_v4_1_debugmodel_multimodal()
    assert trainer.dataloader.per_doc_alignment == cr._per_doc_alignment(trainer.model_spec) == 2


def test_attention_metadata_from_packed_positions(tokenizer):
    """The model-owned metadata hook turns the packed positions into the
    document ids, the ragged boundaries and the indexer's selection masks."""
    loader = _loader(tokenizer, packing=4)
    inputs, labels = next(iter(loader))
    extra = {key: value.clone() for key, value in inputs.items()}
    # A vision batch carries no validity mark any more: the hook must not
    # expect one, and nothing may reach the forward that the forward lacks.
    assert "valid_tokens" not in extra

    class _HookOwner:
        # call the real hook through a minimal owner: the hook dispatches to
        # the model's own metadata builder, so attach the actual construction
        # methods and the ratio table its selection-mask precompute reads
        compress_ratios = (2,)
        build_attention_masks = V41Model.build_attention_masks
        get_attention_masks = V41Model.get_attention_masks

    _, _, extra = _HookOwner().build_attention_masks(inputs, labels, extra)

    metadata = extra["attention_masks"]
    starts = (inputs["positions"][0] == 0).nonzero().flatten().to(torch.int32)
    expected_cu = torch.cat((starts, torch.tensor([inputs["positions"].numel()], dtype=torch.int32)))
    torch.testing.assert_close(metadata.cu_seq_q, expected_cu, rtol=0, atol=0)
    doc_ids = torch.cumsum((inputs["positions"] == 0).to(torch.int32), dim=-1) - 1
    torch.testing.assert_close(metadata.doc_ids_BL, doc_ids, rtol=0, atol=0)
    assert metadata.selection_masks.keys() == {2}
