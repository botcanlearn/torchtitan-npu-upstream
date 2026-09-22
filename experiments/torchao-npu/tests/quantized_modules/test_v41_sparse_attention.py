# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torchao_npu.quantized_modules import v41_sparse_attention as sparse_attention


def test_localize_indices_preserves_document_local_entries():
    """Global top-k coordinates outside a query document are invalidated."""
    topk_indices = torch.tensor([[0, 1, 3], [1, 2, 3], [2, 4, 5]], dtype=torch.int64)
    doc_ids = torch.tensor([0, 0, 1], dtype=torch.int64)
    cu_cmp = torch.tensor([0, 2, 5], dtype=torch.int64)

    localized = sparse_attention._localize_indices(topk_indices, doc_ids, cu_cmp)

    expected = torch.tensor([[0, 1, -1], [1, -1, -1], [0, 2, -1]], dtype=torch.int64)
    assert torch.equal(localized, expected)


def _fake_mla_apply(captured):
    def apply(q, swa_k, cmp_k, indices, sinks, cu_q, cu_cmp, remainder, scale, ratio, window):
        captured.update(
            q=q,
            swa_k=swa_k,
            cmp_k=cmp_k,
            indices=indices,
            sinks=sinks,
            cu_q=cu_q,
            cu_cmp=cu_cmp,
            remainder=remainder,
            scale=scale,
            ratio=ratio,
            window=window,
        )
        lse = torch.zeros((1, q.shape[0], q.shape[1]), dtype=q.dtype, device=q.device)
        return q, lse

    return staticmethod(apply)


@pytest.mark.parametrize(
    ("ratio", "with_shared", "cmp_tokens"),
    [(0, False, 0), (1, True, 4), (2, True, 2)],
    ids=["window-only", "full-resolution-shared-kv", "compressed-shared-kv"],
)
def test_sparse_mla_quantizes_only_swa_and_preserves_main_kv_in_both_passes(
    monkeypatch, ratio, with_shared, cmp_tokens
):
    """Exercise the custom autograd function around mocked NPU kernel boundaries."""
    device = "cpu"
    q = torch.ones((4, 2, 4), device=device, dtype=torch.bfloat16, requires_grad=True)
    swa_k = torch.ones((4, 1, 4), device=device, dtype=torch.bfloat16, requires_grad=True)
    cmp_k = (
        torch.ones((cmp_tokens, 1, 4), device=device, dtype=torch.bfloat16, requires_grad=True) if with_shared else None
    )
    indices = torch.zeros((4, 1, 2), device=device, dtype=torch.int32) if with_shared else None
    sinks = torch.ones((2,), device=device, dtype=torch.float32, requires_grad=True)
    cu_q = torch.tensor([0, 4], device=device, dtype=torch.int32)
    cu_cmp = torch.tensor([0, cmp_tokens], device=device, dtype=torch.int32) if with_shared else None
    remainder = torch.zeros((1,), device=device, dtype=torch.int32) if ratio > 1 else None
    forward_metadata = torch.tensor([7], device=device, dtype=torch.int32)
    grad_metadata = torch.tensor([9], device=device, dtype=torch.int32)
    metadata_calls = []
    quantize_calls = []
    forward_kwargs = {}
    lse = torch.zeros((1, q.shape[0], q.shape[1]), device=device, dtype=q.dtype)

    def metadata(*args, **kwargs):
        metadata_calls.append((args, kwargs))
        return forward_metadata if len(metadata_calls) == 1 else grad_metadata

    def fake_quantize(kv, quant_group_size, quant_mode):
        quantize_calls.append((kv, quant_group_size, quant_mode))
        return kv + 1

    def sparse_flash_mla(q_input, **kwargs):
        forward_kwargs.update(kwargs)
        return q_input + 1, lse

    def sparse_flash_mla_grad(q_input, grad_output, output, saved_lse, **kwargs):
        assert q_input is q
        assert output.shape == q.shape
        assert saved_lse is lse
        torch.testing.assert_close(grad_output, torch.ones_like(q))
        assert kwargs["ori_kv"] is forward_kwargs["ori_kv"]
        assert kwargs["ori_kv"] is not swa_k
        torch.testing.assert_close(kwargs["ori_kv"], torch.full_like(swa_k, 2))
        assert kwargs["cmp_kv"] is cmp_k
        assert kwargs["cmp_sparse_indices"] is indices
        assert kwargs["sinks"] is sinks
        assert kwargs["metadata"] is grad_metadata
        assert kwargs["cmp_residual_kv"] is remainder
        assert kwargs["softmax_scale"] == 0.5
        assert kwargs["cmp_ratio"] == max(ratio, 1)
        return (
            torch.full_like(q, 2),
            torch.full_like(swa_k, 3),
            torch.full_like(cmp_k, 4) if cmp_k is not None else None,
            torch.full_like(sinks, 5),
            None,
            None,
        )

    monkeypatch.setattr(sparse_attention, "sparse_flash_mla_metadata", metadata)
    monkeypatch.setattr(sparse_attention, "sparse_flash_mla_grad_metadata", metadata)
    monkeypatch.setattr(sparse_attention, "fake_quantize_mx_bf16", fake_quantize)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla", sparse_flash_mla)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", sparse_flash_mla_grad)

    output, returned_lse = sparse_attention._SparseMLA.apply(
        q,
        swa_k,
        cmp_k,
        indices,
        sinks,
        cu_q,
        cu_cmp,
        remainder,
        0.5,
        ratio,
        2,
    )

    assert returned_lse is lse
    assert not returned_lse.requires_grad
    assert len(quantize_calls) == 1
    assert quantize_calls[0][0] is swa_k
    assert quantize_calls[0][1:] == (32, "mxfp8_bf16")
    torch.testing.assert_close(forward_kwargs["ori_kv"], swa_k + 1)
    assert forward_kwargs["cmp_kv"] is cmp_k
    assert forward_kwargs["metadata"] is forward_metadata
    assert forward_kwargs["cmp_sparse_indices"] is indices
    assert forward_kwargs["sinks"] is sinks
    assert forward_kwargs["cmp_residual_kv"] is remainder
    assert len(metadata_calls) == 2

    output.sum().backward()
    torch.testing.assert_close(q.grad, torch.full_like(q, 2))
    torch.testing.assert_close(swa_k.grad, torch.full_like(swa_k, 3))
    torch.testing.assert_close(sinks.grad, torch.full_like(sinks, 5))
    if with_shared:
        torch.testing.assert_close(cmp_k.grad, torch.full_like(cmp_k, 4))


def test_forward_ratio_zero_passes_window_only_inputs_to_mla(monkeypatch):
    captured = {}
    monkeypatch.setattr(sparse_attention._SparseMLA, "apply", _fake_mla_apply(captured))

    device = "npu"
    q = torch.randn((1, 4, 2, 512), device=device, dtype=torch.bfloat16)
    swa_k = torch.randn((1, 4, 512), device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        cu_seq_q=torch.tensor([0, 4], device=device, dtype=torch.int32),
        doc_ids_BL=torch.zeros(4, device=device, dtype=torch.int32),
    )
    module = sparse_attention.QuantV41SparseAttention(
        window_size=3,
        softmax_scale=0.125,
        compress_ratio=0,
    )

    output, teacher_lse = module(
        q,
        swa_k,
        None,
        attention_masks=metadata,
        topk_indices=None,
        attn_sink=torch.zeros(2, device=device, dtype=torch.float32),
        wants_teacher=True,
    )

    assert output.shape == q.shape
    assert teacher_lse.shape == (1, 2, 4)
    assert captured["cmp_k"] is None
    assert captured["indices"] is None
    assert captured["cu_cmp"] is None
    assert captured["ratio"] == 0
    assert captured["window"] == 3


def test_forward_ratio_one_uses_full_resolution_shared_kv_without_residual(monkeypatch):
    captured = {}
    monkeypatch.setattr(sparse_attention._SparseMLA, "apply", _fake_mla_apply(captured))

    device = "npu"
    q = torch.randn((1, 4, 2, 512), device=device, dtype=torch.bfloat16)
    swa_k = torch.randn((1, 4, 512), device=device, dtype=torch.bfloat16)
    cmp_k = torch.randn((1, 4, 512), device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        cu_seq_q=torch.tensor([0, 4], device=device, dtype=torch.int32),
        doc_ids_BL=torch.zeros(4, device=device, dtype=torch.int32),
    )
    module = sparse_attention.QuantV41SparseAttention(
        window_size=3,
        softmax_scale=0.125,
        compress_ratio=1,
    )

    output, teacher_lse = module(
        q,
        swa_k,
        cmp_k,
        attention_masks=metadata,
        topk_indices=torch.zeros((1, 4, 2), device=device, dtype=torch.int64),
        attn_sink=torch.zeros(2, device=device, dtype=torch.float32),
        wants_teacher=True,
    )

    assert output.shape == q.shape
    assert teacher_lse.shape == (1, 2, 4)
    assert captured["cu_cmp"].tolist() == [0, 4]
    assert captured["remainder"] is None
    assert captured["cmp_k"].shape == (4, 1, 512)


def test_forward_ratio_two_localizes_and_compacts_indices(monkeypatch):
    captured = {}
    monkeypatch.setattr(sparse_attention._SparseMLA, "apply", _fake_mla_apply(captured))

    device = "npu"
    q = torch.randn((1, 4, 2, 512), device=device, dtype=torch.bfloat16)
    swa_k = torch.randn((1, 4, 512), device=device, dtype=torch.bfloat16)
    cmp_k = torch.randn((1, 2, 512), device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        cu_seq_q=torch.tensor([0, 2, 4], device=device, dtype=torch.int32),
        doc_ids_BL=torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32),
    )
    topk_indices = torch.tensor(
        [[[0, 1], [1, 0], [1, 0], [0, 1]]],
        device=device,
        dtype=torch.int64,
    )
    module = sparse_attention.QuantV41SparseAttention(
        window_size=5,
        softmax_scale=0.25,
        compress_ratio=2,
    )

    output, teacher_lse = module(
        q,
        swa_k,
        cmp_k,
        attention_masks=metadata,
        topk_indices=topk_indices,
        attn_sink=torch.zeros(2, device=device, dtype=torch.float32),
        wants_teacher=True,
    )

    assert output.shape == q.shape
    assert teacher_lse.shape == (1, 2, 4)
    assert captured["cu_cmp"].tolist() == [0, 1, 2]
    assert captured["remainder"].tolist() == [0, 0]
    assert captured["indices"].shape == (4, 1, 2)
    assert captured["indices"].tolist() == [[[0, -1]], [[0, -1]], [[0, -1]], [[0, -1]]]
    assert captured["cmp_k"].shape == (2, 1, 512)
    assert captured["ratio"] == 2
