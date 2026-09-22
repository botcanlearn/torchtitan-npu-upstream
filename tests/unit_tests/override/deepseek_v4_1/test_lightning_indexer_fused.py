# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused V4.1 LightningIndexer: selection parity, carrier gradients, dispatch.

The NPU kernels are replaced by eager math with their exact contract
(document-local indices, ``-1``/``-inf`` padding, the SLIKG closed form
``dI = Z * Y - p``), so these tests pin the wiring: the fused score-and-select
matches the reference path's selection and scores, the gradient carrier
delivers the teacher to the fused backward, the distillation loss logs the
reference value, and every layer -- pool-configured ones included -- runs the
fused path.
"""

import importlib

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.config import derive

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v4_1.indexer import HierarchicalIndexer, IndexerDistillLoss
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import lightning_indexer as li_module
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.lightning_indexer import (
    AscScoreAndSelect,
)

_TOKENS = torch.cat([torch.arange(64), torch.arange(63, -1, -1)]).unsqueeze(0)
_POSITIONS = torch.cat([torch.arange(64), torch.arange(64)]).unsqueeze(0)


# ---------------------------------------------------------------------------
# Eager stand-ins for the NPU kernels, following the verified op contracts.
# ---------------------------------------------------------------------------


def _metadata(positions_1d: torch.Tensor):
    """Build the model's varlen metadata for a packed two-document batch."""
    from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model

    class _ModelOwner:
        compress_ratios = (1, 2)
        get_attention_masks = DeepSeekV41Model.get_attention_masks

    return _ModelOwner().get_attention_masks(positions_1d.unsqueeze(0))


def _fake_li_metadata(*args, **kwargs):
    return torch.zeros(1024, dtype=torch.int32)


def _fake_slig_metadata(*args, **kwargs):
    return torch.zeros(64, dtype=torch.int32)


def _fake_lightning_indexer(
    q,
    k,
    w,
    topk,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    cmp_residual_k,
    metadata,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    **_,
):
    """The LI v2 forward: doc-local top-k of the relaxed score, -1/-inf padded.

    The relaxed score and the selection run through the reference path's
    exact expressions over the packed batch: a per-doc contraction or a
    doc-sliced top-k rounds and tie-breaks differently (relu zeroes tie at
    the top-k boundary), which the parity tests must not mistake for a
    wiring difference.  The picks come back in document-local coordinates
    with -1/-inf padding, the kernel's contract.
    """
    total = q.shape[0]
    indices = torch.full((total, 1, topk), -1, dtype=torch.int32)
    values = torch.full((total, 1, topk), float("-inf"))
    scores_blhn = torch.einsum("blhd,bnd->blhn", q.unsqueeze(0), k.squeeze(1).unsqueeze(0))
    scores_bln = (scores_blhn.relu() * w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)
    cu_q = cu_seqlens_q.to(torch.long).tolist()
    cu_k = cu_seqlens_k.to(torch.long).tolist()
    visible = torch.zeros_like(scores_bln, dtype=torch.bool)
    starts = torch.zeros(total, dtype=torch.long)
    for doc in range(len(cu_q) - 1):
        q0, q1 = cu_q[doc], cu_q[doc + 1]
        k0, k1 = cu_k[doc], cu_k[doc + 1]
        token_pos = torch.arange(q1 - q0)
        visible[0, q0:q1, k0:k1] = (
            torch.arange(k1 - k0).view(1, -1) < ((token_pos + 1) // cmp_ratio).view(-1, 1)
        )
        starts[q0:q1] = k0
    relaxed = scores_bln.masked_fill(~visible, float("-inf"))
    pick = min(topk, relaxed.size(-1))
    selected = relaxed.topk(pick, dim=-1, sorted=False).indices
    selected = selected.topk(pick, dim=-1, largest=False, sorted=True).values[0]
    reachable = visible[0].gather(-1, selected)
    local = torch.where(reachable, selected - starts.unsqueeze(-1), torch.full_like(selected, -1))
    indices[:, 0, :pick] = local.to(torch.int32)
    values[:, 0, :pick] = relaxed[0].gather(-1, selected)
    return indices, values


def _fake_slig(
    q,
    k,
    w,
    sparse_indices,
    attn_softmax_l1_norm,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    cmp_residual_k,
    metadata,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    **_,
):
    """The SLIKG backward: the closed form dI = Z * Y - p of the reference loss."""
    # The real Function's backward runs grad-disabled; the stand-in rebuilds
    # the eager graph internally, so it opts back in.
    with torch.enable_grad():
        q_e = q.detach().float().requires_grad_(True)
        k_e = k.detach().float().requires_grad_(True)
        w_e = w.detach().requires_grad_(True)
        si = sparse_indices.squeeze(1).to(torch.long)
        p = attn_softmax_l1_norm.squeeze(1)
        lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
        offsets = torch.repeat_interleave(cu_seqlens_k[:-1].long(), lengths)
        sel = (si + offsets[:, None]).clamp_min(0)
        valid = (si >= 0).float()
        k_sel = k_e.squeeze(1)[sel]
        logits = torch.einsum("thd,tkd->tkh", q_e, k_sel).relu() * w_e.unsqueeze(1)
        i_sel = logits.sum(-1) * valid
        i_masked = torch.where(valid > 0, i_sel, torch.full_like(i_sel, float("-inf")))
        z = p.sum(-1)
        loss = (z * torch.logsumexp(i_masked, -1).nan_to_num(0.0) - (p * i_sel).sum(-1)).sum()
        dq, dk, dw = torch.autograd.grad(loss, [q_e, k_e, w_e])
        softmax_out = torch.softmax(i_masked, -1).nan_to_num(0.0)
    return dq.to(q.dtype), dk.to(k.dtype), dw, softmax_out.unsqueeze(1)


@pytest.fixture(autouse=True)
def _fake_kernels(monkeypatch):
    monkeypatch.setattr(li_module, "lightning_indexer_metadata", _fake_li_metadata)
    monkeypatch.setattr(li_module, "sparse_lightning_indexer_kl_loss_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer", _fake_lightning_indexer)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", _fake_slig)


def _tiny_model(monkeypatch, *, fused: bool, smla: bool = False, pools: bool = True):
    """The registered debug flavor at CPU-sized widths, optionally with the LI override."""
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    make_config = registry._make_v41_config

    def tiny_config(**kwargs):
        kwargs.update(
            dim=8,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            n_groups=1,
            index_n_heads=2,
            index_head_dim=4,
            moe_inter_dim=16,
            vision_dim=8,
            vision_heads=2,
            vision_inter_dim=16,
        )
        return make_config(**kwargs)

    monkeypatch.setattr(registry, "_make_v41_config", tiny_config)
    config = registry.model_registry("deepseek_v4_1_debugmodel").model
    config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = 64
    if not pools:
        for _, cfg, _, _ in config.traverse(HierarchicalIndexer.Config):
            cfg.score_and_select.candidate_topk_blocks = 0
            cfg.score_and_select.candidate_block_size = 0
    if fused:
        for _, cfg, parent, attr in config.traverse(HierarchicalIndexer.Config):
            cfg.score_and_select = derive(cfg.score_and_select, AscScoreAndSelect.Config)
    if smla:
        from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention

        for layer in config.layers:
            layer.attention.inner_attention = derive(layer.attention.inner_attention, AscV41SparseAttention.Config)
    for _, loss_cfg, _, _ in config.traverse(IndexerDistillLoss.Config):
        loss_cfg.global_batch_size = 1
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config)
    return model


def _run_forward_backward(model):
    _, _, kwargs = model.build_attention_masks(_TOKENS, _TOKENS, {"positions": _POSITIONS})
    output = model(_TOKENS, **kwargs)
    output.sum().backward()
    return output


def test_pool_free_indexers_match_the_reference_end_to_end(monkeypatch):
    """The fused path reproduces the pool-free reference model's output, loss and gradients.

    With the kernels faked to the reference math, the only structural
    difference is the loss's student term (the linear carrier instead of the
    log-softmax), which the SLIKG closed form matches exactly -- so outputs,
    the logged distillation value, and every indexer's gradients agree with
    the eager model.  The candidate pool is not part of the fused path: every
    layer selects over all visible entries, so the reference is the same
    eager model with pools disabled.
    """
    eager = _tiny_model(monkeypatch, fused=False, pools=False)
    fused = _tiny_model(monkeypatch, fused=True)

    # Every layer runs the fused node, the pool source and its searchers
    # included: the candidate pool is bypassed, so the eager reference is
    # the same model with pools disabled.
    fused_sources = [fused.layers[k].attention.indexer.score_and_select for k in ("2", "8", "14")]
    fused_pool = [fused.layers[k].attention.indexer.score_and_select for k in ("20", "24", "28")]
    assert all(isinstance(m, AscScoreAndSelect) for m in fused_sources)
    assert all(isinstance(m, AscScoreAndSelect) for m in fused_pool)

    out_eager = _run_forward_backward(eager)
    out_fused = _run_forward_backward(fused)
    torch.testing.assert_close(out_fused, out_eager, rtol=1e-5, atol=1e-6)

    for layer in ("2", "8", "14", "20", "24", "28"):
        loss_e = eager.layers[layer].attention.inner_attention.aux_loss
        loss_f = fused.layers[layer].attention.inner_attention.aux_loss
        torch.testing.assert_close(loss_f.read(), loss_e.read(), rtol=1e-5, atol=1e-6)
        # Only the params the layer's mode owns: Full Mode layers carry the
        # key projections, Reindex Mode layers only their query and weights.
        for name, param_f in fused.layers[layer].attention.indexer.named_parameters():
            param_e = eager.layers[layer].attention.indexer.get_parameter(name)
            assert param_e.grad is not None and param_f.grad is not None, (layer, name)
            torch.testing.assert_close(param_f.grad, param_e.grad, rtol=1e-4, atol=1e-6)


def test_fused_selection_matches_the_reference_score_and_select(monkeypatch):
    """Per-layer parity: same valid index sets, same scores, kernel-shaped outputs."""
    eager = _tiny_model(monkeypatch, fused=False)
    fused = _tiny_model(monkeypatch, fused=True)
    _, _, kwargs = eager.build_attention_masks(_TOKENS, _TOKENS, {"positions": _POSITIONS})
    metadata = kwargs["attention_masks"]

    eager_indexer = eager.layers["2"].attention.indexer.score_and_select
    fused_indexer = fused.layers["2"].attention.indexer.score_and_select
    torch.manual_seed(7)
    b, l = 1, 128
    hi, di = eager_indexer.num_index_heads, eager_indexer.index_head_dim
    idx_q = torch.randn(b, l, hi, di, requires_grad=True)
    idx_k = torch.randn(b, 64, di, requires_grad=True)
    weights = torch.randn(b, l, hi, requires_grad=True)

    ei, es, _ = eager_indexer._score_and_select(idx_q, idx_k, weights, metadata, candidates_BLN=None)
    fi, fs, _ = fused_indexer._score_and_select(idx_q, idx_k, weights, metadata, candidates_BLN=None)

    assert fi.dtype == torch.long
    # The kernel contract's float32 scores, unlike the eager model-dtype logits.
    assert fs.dtype == torch.float32
    # Same reachable selection per row (the fake emits the reference's
    # ascending order; the real kernel's order is unspecified), and the
    # fused scores are the reference scores at the same entries.
    for t in range(l):
        eager_row = {int(j): float(v) for j, v in zip(ei[0, t].tolist(), es[0, t].tolist()) if j >= 0}
        fused_row = {int(j): float(v) for j, v in zip(fi[0, t].tolist(), fs[0, t].tolist()) if j >= 0}
        assert fused_row.keys() == eager_row.keys(), t
        for entry, value in fused_row.items():
            assert abs(value - eager_row[entry]) < 1e-5, (t, entry)
    # The marks agree slot by slot: an unreachable slot is -1 in the indices
    # and -inf in the scores.
    assert torch.equal(torch.isfinite(fs), fi >= 0)

    # The clamp: with a pool smaller than index_topk (seq 512 packs 256
    # entries per ratio-2 layer while the kernel topk is pinned to 512), the
    # fused path must return the eager path's K and the same selection.  A
    # fresh, self-consistent two-document batch keeps the precomputed
    # selection masks matching the smaller pool.
    small_metadata = _metadata(torch.cat([torch.arange(16), torch.arange(16)]))
    idx_q2 = torch.randn(1, 32, hi, di, requires_grad=True)
    idx_k2 = torch.randn(1, 16, di, requires_grad=True)
    weights2 = torch.randn(1, 32, hi, requires_grad=True)
    ei2, es2, _ = eager_indexer._score_and_select(idx_q2, idx_k2, weights2, small_metadata, candidates_BLN=None)
    fi2, fs2, _ = fused_indexer._score_and_select(idx_q2, idx_k2, weights2, small_metadata, candidates_BLN=None)
    assert fi2.shape[-1] == ei2.shape[-1] == 16
    assert fs2.dtype == torch.float32
    for t in range(32):
        assert set(fi2[0, t][fi2[0, t] >= 0].tolist()) == set(ei2[0, t][ei2[0, t] >= 0].tolist()), t
    assert torch.equal(torch.isfinite(fs2), fi2 >= 0)


def test_carrier_gradient_is_the_teacher_and_the_value_matches(monkeypatch):
    """The fused loss hands the teacher to the backward and logs the reference value."""
    loss = build_cpu_model(
        IndexerDistillLoss.Config(
            coeff=2.0,
            reduce_mesh="batch",
            global_batch_size=1,
            softmax_scale=1.0,
        )
    )
    q_BLHD = torch.zeros(1, 1, 1, 1)
    cmp_k_BND = torch.zeros(1, 2, 1)
    topk_indices_BLK = torch.tensor([[[0, 1]]])
    log_two = float(torch.log(torch.tensor(2.0)))
    lse_BLH = torch.full((1, 1, 1), log_two)
    # The carrier: the layer's indexer reported the kernel path (wired onto
    # the loss at build time), and the scores are the kernel-contract float32.
    loss.score_gradient = "teacher"
    topk_scores = torch.tensor([[[float(torch.log(torch.tensor(3.0))), 0.0]]], requires_grad=True)
    carrier = torch.zeros(1, 1, 1)

    returned = loss(q_BLHD, cmp_k_BND, topk_indices_BLK, lse_BLH, topk_scores, carrier=carrier)
    torch.testing.assert_close(returned, carrier, rtol=0, atol=0)
    assert returned.grad_fn is not None
    returned.sum().backward()

    # Teacher p = [0.5, 0.5] and coeff * scale = 2.0: the carrier's gradient
    # is the (scaled) teacher the SLIKG kernel consumes.
    torch.testing.assert_close(topk_scores.grad, torch.full((1, 1, 2), 1.0), rtol=1e-6, atol=1e-7)
    # The logged value is the reference KL: log 2 - 0.5 log 3.
    torch.testing.assert_close(
        loss.read(), torch.tensor(log_two - 0.5 * float(torch.log(torch.tensor(3.0)))), rtol=1e-6, atol=1e-7
    )

    # A pool group (the flag clear) keeps the parent path: dI = Z * Y - p.
    loss.score_gradient = "logits"
    eager_scores = torch.tensor(
        [[[float(torch.log(torch.tensor(3.0))), 0.0]]], dtype=torch.bfloat16, requires_grad=True
    )
    loss(q_BLHD, cmp_k_BND, topk_indices_BLK, lse_BLH, eager_scores, carrier=carrier).sum().backward()
    torch.testing.assert_close(
        eager_scores.grad, torch.tensor([[[0.25, -0.25]]], dtype=torch.bfloat16) * 2.0, rtol=1e-2, atol=1e-2
    )

    # Ordinary logits use their mathematical gradient, including float32 scores.
    loss.score_gradient = "logits"
    plain_scores = torch.tensor([[[float(torch.log(torch.tensor(3.0))), 0.0]]], requires_grad=True)
    loss(q_BLHD, cmp_k_BND, topk_indices_BLK, lse_BLH, plain_scores, carrier=carrier).sum().backward()
    torch.testing.assert_close(plain_scores.grad, torch.tensor([[[0.25, -0.25]]]) * 2.0, rtol=1e-6, atol=1e-7)


def test_asc_li_stack_applies_without_nested_claim_conflicts(monkeypatch):
    """The fused node override composes with the blanket norm/rope stacks.

    ``asc_li`` swaps only the parameterless ``ScoreAndSelect`` node, which
    owns no nested norm or rope, so the blanket ``common.rms_norm.asc`` and
    ``common.rope.asc_complex`` apply alongside it with disjoint claims: the
    indexer's ``k_norm`` and ``rope`` are covered like every other site.
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides
    from torchtitan.models.common.rope import ComplexRoPE

    from torchtitan_npu.override.common.rms_norm import AscRMSNorm
    from torchtitan_npu.override.common.rope import AscComplexRoPE

    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    model = registry.model_registry("deepseek_v4_1_debugmodel").model

    apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.rms_norm.asc",
                "torchtitan_npu.override.common.rope.asc_complex",
                "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc_li",
                "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
            ]
        ),
        model,
    )

    indexer = model.layers[2].attention.indexer
    assert isinstance(indexer.score_and_select, AscScoreAndSelect.Config)
    assert isinstance(indexer.rope, ComplexRoPE.Config)
    assert isinstance(model.layers[0].attention.q_norm, AscRMSNorm.Config)
    assert isinstance(model.layers[0].attention_norm, AscRMSNorm.Config)
    assert isinstance(model.layers[0].attention.rope, AscComplexRoPE.Config)


def test_full_model_smla_has_no_teacher_reconstruction(monkeypatch):
    from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc

    model = _tiny_model(monkeypatch, fused=True, smla=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)

    def forbidden(*args, **kwargs):
        pytest.fail("SMLA must not reconstruct the teacher")

    monkeypatch.setattr(IndexerDistillLoss, "_teacher", forbidden)
    monkeypatch.setattr(ascendc, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(ascendc, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kw: (q.clone(), q.new_zeros(1, q.shape[0], q.shape[1])),
    )

    def backward(q, grad, *args, **kw):
        indices = kw["cmp_sparse_indices"]
        teacher = (indices >= 0).float() * 0.01 if indices is not None else None
        shared = kw["cmp_kv"]
        return (
            grad,
            torch.zeros_like(kw["ori_kv"]),
            torch.zeros_like(shared) if shared is not None else None,
            torch.zeros_like(kw["sinks"]),
            None,
            teacher,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", backward)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run_forward_backward(model)
    for layer_id in ("2", "3", "8", "14", "20", "21", "24", "28"):
        inner = model.layers[layer_id].attention.inner_attention
        # Every layer -- the pool group included -- hands the teacher to the
        # fused backward.
        assert inner.score_gradient == "teacher"
        assert torch.isfinite(inner.aux_loss.read())
    for name, parameter in model.named_parameters():
        if ".indexer." in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
