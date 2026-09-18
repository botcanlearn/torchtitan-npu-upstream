# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused sparse adapter's coordinate translation and gradient plumbing."""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.model import V41Model
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc

from tests.unit_tests.models.mtp_test_utils import build_cpu_model


def _metadata(positions_1d: torch.Tensor):
    """Build the metadata through the model's own construction.

    ``get_attention_masks`` only reads the model's ratio table off the
    instance (its selection-mask precompute), so a minimal owner carrying
    the ratios exercises the real logic instead of re-deriving it here.
    """

    class _ModelOwner:
        compress_ratios = (1, 2)
        get_attention_masks = V41Model.get_attention_masks

    return _ModelOwner().get_attention_masks(positions_1d.unsqueeze(0))


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("bounds", [[0, 8], [0, 4, 8]], ids=["single-doc", "two-docs"])
def test_kernel_receives_document_local_indices(monkeypatch, ratio, bounds):
    length = bounds[-1]
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)])
    metadata = _metadata(positions)
    assert torch.equal(metadata.cu_seq_q, torch.tensor(bounds, dtype=torch.int32))
    # Per-document alignment makes every document's compressed length exact.
    cu_cmp = metadata.cu_seq_q // ratio

    # Model indices are global compressed-pool coordinates, -1 for unused
    # slots: another document's entry, the first/last valid entry, an unused
    # slot and an exclusive-end index.
    indices = torch.empty((1, length, 5), dtype=torch.int32)
    expected = torch.empty((length, 1, 5), dtype=torch.int32)
    for doc, (begin, end) in enumerate(pairwise(bounds)):
        start, stop = int(cu_cmp[doc]), int(cu_cmp[doc + 1])
        foreign = 0 if start else int(cu_cmp[-1])
        indices[0, begin:end] = torch.tensor([foreign, start, -1, stop - 1, stop])
        last = stop - start - 1
        # Invalid slots compact stably to the back; valid keys keep their order.
        expected[begin:end, 0] = torch.tensor([0, last, -1, -1, -1])

    calls = []

    def kernel(q, original, shared, local_indices, *args):
        calls.append(local_indices)
        return torch.zeros_like(q), torch.zeros((1, q.shape[0], q.shape[1]))

    monkeypatch.setattr(ascendc._SparseMLA, "apply", kernel)
    attention = ascendc.AscV41SparseAttention.Config(
        window_size=2, softmax_scale=0.5, compress_ratio=ratio, aux_loss=None
    ).build()
    q = torch.zeros((1, length, 2, 4), dtype=torch.bfloat16)
    topk_before = indices.clone()
    out, lse = attention._compute_attention(
        q,
        torch.zeros((1, length, 4), dtype=torch.bfloat16),
        torch.zeros((1, int(cu_cmp[-1]), 4), dtype=torch.bfloat16),
        attention_masks=metadata,
        topk_indices=indices,
        attn_sink=torch.zeros(2),
        wants_teacher=True,
    )
    assert out.shape == q.shape
    # The kernel LSE is [1, S, H]; the teacher seam receives [B, H, L].
    assert lse.shape == (1, 2, length)
    assert len(calls) == 1
    assert calls[0].is_contiguous()
    torch.testing.assert_close(calls[0], expected, rtol=0, atol=0)
    # The model's global indices are consumed, never rewritten in place.
    torch.testing.assert_close(indices, topk_before, rtol=0, atol=0)


def test_forward_feeds_the_kernel_lse_to_the_distillation_loss(monkeypatch):
    """The fused port keeps the parent forward's loss wiring: the teacher is
    rebuilt from the kernel's full-softmax LSE and the student logits train."""
    from torchtitan_npu.models.deepseek_v4_1.indexer import IndexerDistillLoss

    def kernel(q, original, shared, local_indices, *args):
        seq, heads = q.shape[0], q.shape[1]
        lse = torch.full((1, seq, heads), float(torch.log(torch.tensor(2.0))))
        return torch.zeros_like(q), lse

    monkeypatch.setattr(ascendc._SparseMLA, "apply", kernel)

    attention = build_cpu_model(
        ascendc.AscV41SparseAttention.Config(
            window_size=2,
            softmax_scale=0.5,
            compress_ratio=2,
            aux_loss=IndexerDistillLoss.Config(
                coeff=1.0,
                reduce_mesh="batch",
                global_batch_size=1,
                softmax_scale=0.5,
            ),
        )
    )
    metadata = _metadata(torch.arange(8))
    q = torch.zeros(1, 8, 1, 4, dtype=torch.bfloat16)
    topk_indices = torch.tensor([[[0, 1]] * 8])
    topk_scores = torch.tensor(
        [[[float(torch.log(torch.tensor(3.0))), 0.0]] * 8], requires_grad=True
    )
    out = attention(
        q,
        torch.zeros(1, 8, 4, dtype=torch.bfloat16),
        torch.zeros(1, 4, 4, dtype=torch.bfloat16),
        attention_masks=metadata,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        attn_sink=torch.zeros(1),
    )
    out.sum().backward()
    # Uniform teacher p = [0.5, 0.5] with Z = 1 and student Y = [0.75, 0.25]:
    # dI = Z * Y - p = [0.25, -0.25].
    torch.testing.assert_close(
        topk_scores.grad,
        torch.tensor([[[0.25, -0.25]] * 8]),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("with_shared", [False, True], ids=["window", "shared-kv"])
def test_backward_routes_kernel_gradients_and_ignores_the_lse_gradient(monkeypatch, with_shared):
    q = torch.ones(4, 2, 4, requires_grad=True)
    original = torch.ones(4, 1, 4, requires_grad=True)
    shared = torch.ones(2, 1, 4, requires_grad=True) if with_shared else None
    sink = torch.ones(2, requires_grad=True)
    indices = torch.zeros(4, 1, 2, dtype=torch.int32) if with_shared else None
    cu = torch.tensor([0, 4], dtype=torch.int32)
    cmp_cu = torch.tensor([0, 2], dtype=torch.int32) if with_shared else None
    remainder = torch.zeros(1, dtype=torch.int32) if with_shared else None
    meta, lse = torch.tensor([7]), torch.zeros(4, 2)
    forward_args = {}
    monkeypatch.setattr(ascendc, "sparse_flash_mla_metadata", lambda *a, **kw: meta)
    monkeypatch.setattr(ascendc, "sparse_flash_mla_grad_metadata", lambda *a, **kw: meta)

    def forward(q, **kwargs):
        forward_args.update(kwargs)
        return q.clone(), lse

    def backward(q_in, grad, output, saved_lse, **kwargs):
        assert q_in is q and saved_lse is lse
        torch.testing.assert_close(grad, torch.ones_like(q))
        for name in ("ori_kv", "cmp_kv", "cmp_sparse_indices", "sinks", "metadata",
                     "cu_seqlens_q", "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "cmp_residual_kv"):
            assert kwargs[name] is forward_args[name]
        assert kwargs["softmax_scale"] == 0.5
        assert kwargs["cmp_ratio"] == (2 if with_shared else 1)
        assert kwargs["ori_win_left"] == 1
        return (torch.full_like(q, 2), torch.full_like(original, 3),
                torch.full_like(shared, 5) if with_shared else None,
                torch.full_like(sink, 7), None, None)

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla", forward)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", backward)
    out, returned_lse = ascendc._SparseMLA.apply(
        q, original, shared, indices, sink, cu, cmp_cu, remainder, 0.5, 2 if with_shared else 0, 2
    )
    # The LSE passes through as a detached teacher signal.
    assert returned_lse is lse
    assert not returned_lse.requires_grad
    out.sum().backward()
    for value, expected in ((q, 2), (original, 3), (sink, 7)):
        torch.testing.assert_close(value.grad, torch.full_like(value, expected))
    if with_shared:
        torch.testing.assert_close(shared.grad, torch.full_like(shared, 5))
