# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused V4.1 LightningIndexer: selection parity, teacher edge, dispatch.

The NPU kernels are replaced by eager math with their exact contract
(document-local indices, ``-1``/``-inf`` padding, the SLIKG closed form
``dI = Z * Y - p``), so these tests pin the wiring: the fused selector matches the
reference path's selection and scores, the teacher reaches SLIKG on the ``topk_scores``
edge, and every layer -- pool-configured ones included -- runs the fused path.

**The indexer only learns when both overrides are enabled together.** ``sparse_attn.asc``
produces the teacher (SMLAG's backward is the only source of it) and
``lightning_indexer.asc`` consumes it (SLIKG is the only consumer). With the selector
override alone the indexer's selection feeds the attention as indices, which carry no
gradient, so its parameters stay frozen; that is why nothing here checks an "eager
indexer gradient" -- with the distillation loss removed there is no longer a path to one.
"""

from itertools import pairwise
import importlib
import types

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.config import derive

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v4_1.indexer import HierarchicalIndexer, Selector
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer import ascendc as li_module
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer.ascendc import (
    AscSelector,
)

_TOKENS = torch.cat([torch.arange(64), torch.arange(63, -1, -1)]).unsqueeze(0)
_POSITIONS = torch.cat([torch.arange(64), torch.arange(64)]).unsqueeze(0)


# ---------------------------------------------------------------------------
# Eager stand-ins for the NPU kernels, following the verified op contracts.
# ---------------------------------------------------------------------------


def _metadata(bounds: list[int], *, compress_ratios: tuple[int, ...] = (1, 2)):
    """Build the model's varlen metadata for a packed row of these document boundaries.

    The positions that produce the boundaries are rebuilt here, so the metadata always
    comes from the model's own builder; one ``arange`` per document is what a packed
    loader emits.
    """
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)

    from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model

    class _ModelOwner:
        pass

    owner = _ModelOwner()
    owner.compress_ratios = compress_ratios
    return DeepSeekV41Model.get_attention_masks(owner, positions)


def _fake_li_metadata(*args, **kwargs):
    return torch.zeros(1024, dtype=torch.int32)


def _fake_slig_metadata(*args, **kwargs):
    return torch.zeros(64, dtype=torch.int32)


def _teacher_sources(index_source_layers: tuple[int, ...]) -> tuple[int, ...]:
    """The layers whose selection the teacher edge trains -- one SLIKG call each.

    Every index source owns a selection, and a selection only reaches SLIKG through the
    layer that owns it, so the sources are exactly the calls.  The count therefore comes
    from the layer table rather than from the run.
    """
    return tuple(index_source_layers)


def _relaxed_scores(q, k, w, cu_seqlens_q, cu_seqlens_k, cmp_ratio):
    """The relaxed index score over the packed batch, and the visibility that masks it.

    The reference path's exact expressions: a per-doc contraction or a doc-sliced top-k
    rounds and tie-breaks differently (relu zeroes tie at the top-k boundary), which the
    parity tests must not mistake for a wiring difference.  ``starts`` is where each row's
    document begins on the compressed axis, which is the origin of its document-local
    coordinates.
    """
    total = q.shape[0]
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
        visible[0, q0:q1, k0:k1] = torch.arange(k1 - k0).view(1, -1) < ((token_pos + 1) // cmp_ratio).view(-1, 1)
        starts[q0:q1] = k0
    return scores_bln.masked_fill(~visible, float("-inf")), visible, starts


def _select(relaxed, reachable, starts, topk):
    """Doc-local top-k of the relaxed score, ``-1``/``-inf`` padded.

    The picks come back in document-local coordinates, the kernels' contract, and the
    unreachable ones as ``-1``.
    """
    total = relaxed.size(1)
    indices = torch.full((total, 1, topk), -1, dtype=torch.int32)
    values = torch.full((total, 1, topk), float("-inf"))
    pick = min(topk, relaxed.size(-1))
    selected = relaxed.topk(pick, dim=-1, sorted=False).indices
    selected = selected.topk(pick, dim=-1, largest=False, sorted=True).values[0]
    ok = reachable[0].gather(-1, selected)
    local = torch.where(ok, selected - starts.unsqueeze(-1), torch.full_like(selected, -1))
    indices[:, 0, :pick] = local.to(torch.int32)
    values[:, 0, :pick] = relaxed[0].gather(-1, selected)
    return indices, values


def _candidate_pool(relaxed, visible, starts, capacity, block_size):
    """The kernels' candidate pool: best blocks by max score, pinned newest, ``-1`` tail.

    This follows the *kernel's* rule, which is not the reference's: the newest block is
    pinned only when it is partly filled.  ``select_candidate_blocks`` pins it
    unconditionally, so the two pools differ whenever the visible prefix ends exactly on a
    block boundary.  The kernel and the vendor's own CPU golden both take this side.
    """
    total = relaxed.size(1)
    block_scores = relaxed.unflatten(-1, (-1, block_size)).amax(dim=-1)
    block_indices = torch.full((total, 1, capacity), -1, dtype=torch.int32)
    block_length = torch.zeros((total, 1), dtype=torch.int32)
    for row in range(total):
        visible_entries = int(visible[0, row].sum())
        if visible_entries == 0:
            continue
        first = int(starts[row]) // block_size
        reachable = (visible_entries + block_size - 1) // block_size
        window = block_scores[0, row, first : first + reachable].clone()
        if visible_entries % block_size != 0:
            window[-1] = float("inf")
        length = min(reachable, capacity)
        picked = window.topk(length).indices
        block_indices[row, 0, :length] = picked.to(torch.int32)
        block_length[row, 0] = length
    return block_indices, block_length


def _fake_qli(
    q,
    k,
    w,
    descale_q,
    descale_k,
    topk,
    quant_mode,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    cmp_residual_k,
    metadata,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
    **_,
):
    """The QLI forward: doc-local top-k of the relaxed score, plus the candidate pool.

    With the pool disabled the operator returns both pool outputs as 1-D empties rather
    than as shaped-but-empty tensors, which the caller never reads but the stand-in has to
    reproduce to stay honest about the ABI.
    """
    relaxed, visible, starts = _relaxed_scores(q, k, w, cu_seqlens_q, cu_seqlens_k, cmp_ratio)
    indices, values = _select(relaxed, visible, starts, topk)
    if candidate_topk_blocks <= 0:
        empty = torch.empty((0,), dtype=torch.int32)
        return indices, values, empty, empty
    block_indices, block_length = _candidate_pool(
        relaxed, visible, starts, candidate_topk_blocks, candidate_block_size
    )
    return indices, values, block_indices, block_length


def _fake_qsli(
    q,
    k,
    w,
    descale_q,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    candidate_block_size,
    *,
    descale_k=None,
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
    """The QSLI forward: the same selection, restricted to the pool's blocks.

    The length is authoritative and the table's tail is never read -- the operator ignores
    it by contract -- so the stand-in reads exactly the prefix the length names.
    """
    relaxed, visible, starts = _relaxed_scores(q, k, w, cu_seqlens_q, cu_seqlens_k, cmp_ratio)
    in_pool = torch.zeros_like(visible)
    for row in range(q.shape[0]):
        length = int(candidate_block_length[row, 0])
        for block in candidate_block_indices[row, 0, :length].tolist():
            begin = int(starts[row]) + block * candidate_block_size
            in_pool[0, row, begin : begin + candidate_block_size] = True
    reachable = visible & in_pool
    return _select(relaxed.masked_fill(~reachable, float("-inf")), reachable, starts, topk)


def _fake_legacy_li(
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
    """The non-quantized ``lightning_indexer``: the same selection, no pool.

    It reads bf16 operands directly -- nothing packs them -- and returns two tensors, which
    is the whole of what the legacy switch changes about the selection.
    """
    relaxed, visible, starts = _relaxed_scores(q, k, w, cu_seqlens_q, cu_seqlens_k, cmp_ratio)
    return _select(relaxed, visible, starts, topk)


def _fake_pack_mxfp4(x):
    """Bypass the MXFP4 round trip: the stand-ins read the bf16 operand directly.

    Packing is a storage detail these fakes do not model -- their contract is the selection
    and the pool, not the quantization -- so the operand passes through and the descale,
    which nothing here reads, is dropped.
    """
    return x.contiguous(), None


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
    # Every operator this module drives is replaced below, so the unit-test environment does
    # not need any of them to exist.  That matters most for ``ds41``: it ships in a separate
    # kernel package the environment need not carry, and reaching for the namespace would
    # raise in this autouse fixture -- turning a missing package into a collection error for
    # the whole module rather than into the stand-ins it was always going to use.
    #
    # The stand-ins replace the quantized pair and the packing between them; SLIKG and its
    # metadata keep their unquantized names, and the legacy operator gets its own pair so a
    # test can tell which path ran.
    ds41 = types.SimpleNamespace(
        quant_lightning_indexer_metadata=_fake_li_metadata,
        quant_sparse_lightning_indexer_metadata=_fake_li_metadata,
        quant_lightning_indexer=_fake_qli,
        quant_sparse_lightning_indexer=_fake_qsli,
    )
    # Populated by construction: monkeypatch refuses to create a name that is not already
    # there, so the stand-ins cannot be set one by one onto a fresh namespace.
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "ds41", ds41, raising=False)
    monkeypatch.setattr(li_module, "_pack_mxfp4", _fake_pack_mxfp4)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer_metadata", _fake_li_metadata, raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer", _fake_legacy_li, raising=False)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad_metadata",
        _fake_slig_metadata,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", _fake_slig, raising=False
    )


def _tiny_model(monkeypatch, *, fused: bool, pools: bool = True, legacy: bool = False):
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
            cfg.selector.candidate_topk_blocks = -1
            cfg.selector.candidate_block_size = -1
    if fused:
        # ``fused`` means the fused stack, not one of its halves: the teacher edge only
        # exists when both sides are fused, which the model itself enforces.  Ratio-0
        # layers are left on the reference core because they have no compressed stream.
        from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention

        for layer in config.layers:
            if layer.attention.inner_attention.compress_ratio == 0:
                continue
            layer.attention.indexer.selector = derive(
                layer.attention.indexer.selector, AscSelector.Config, legacy=legacy
            )
            layer.attention.inner_attention = derive(
                layer.attention.inner_attention, AscV41SparseAttention.Config
            )
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config)
    return model


def _run_forward_backward(model):
    _, _, kwargs = model.build_attention_masks(_TOKENS, _TOKENS, {"positions": _POSITIONS})
    output = model(_TOKENS, **kwargs)
    output.sum().backward()
    return output


def test_fused_selection_matches_the_reference_selector(monkeypatch):
    """Per-layer parity: same valid index sets, same scores, kernel-shaped outputs."""
    eager = _tiny_model(monkeypatch, fused=False)
    fused = _tiny_model(monkeypatch, fused=True)
    _, _, kwargs = eager.build_attention_masks(_TOKENS, _TOKENS, {"positions": _POSITIONS})
    metadata = kwargs["attention_masks"]

    eager_indexer = eager.layers["2"].attention.indexer.selector
    fused_indexer = fused.layers["2"].attention.indexer.selector
    torch.manual_seed(7)
    b, l = 1, 128
    hi, di = eager_indexer.num_index_heads, eager_indexer.index_head_dim
    idx_q = torch.randn(b, l, hi, di, requires_grad=True)
    idx_k = torch.randn(b, 64, di, requires_grad=True)
    weights = torch.randn(b, l, hi, requires_grad=True)

    ei, es, _ = eager_indexer.forward(idx_q, idx_k, weights, metadata, candidates_BL1C=None)
    fi, fs, _ = fused_indexer.forward(idx_q, idx_k, weights, metadata, candidates_BL1C=None)

    # int32, the dtype the kernels speak -- the LI op emits it and SMLAG/SLIKG take it,
    # so it is never widened.  The eager selector's int64 is a ``torch.topk`` artefact and
    # the reference casts it down too (``.int()``).
    assert fi.dtype == torch.int32
    # The fused carrier is a fabricated FP32 buffer of the selection's shape.  Its
    # *contents* are meaningless by construction -- the teacher rides its gradient, not
    # its values -- so only shape, dtype and the graph edge are asserted here.
    assert fs.dtype == torch.float32
    assert es.dtype == fs.dtype == torch.float32
    assert fs.shape == fi.shape
    # Same reachable selection per row.
    #
    # The two selectors speak different coordinate systems: the eager one numbers the
    # whole packed pool, the fused one numbers each document's own entries because that
    # is what the kernel grids.  The comparison is exact once the fused indices are
    # shifted by their document's start.
    starts = metadata.kernel.frame_for(eager_indexer.compress_ratio).cu_seqlens.to(torch.long)
    docs = metadata.ref.doc_ids_BL.reshape(-1)
    for t in range(l):
        eager_row = {int(j) for j in ei[0, t].tolist() if j >= 0}
        fused_row = {int(starts[docs[t]]) + int(j) for j in fi[0, t].tolist() if j >= 0}
        assert fused_row == eager_row, t

    # A pool smaller than index_topk: the fused kernel keeps its own K and reports the
    # unreachable slots as ``-1``, so it is wider than the eager selection.  The
    # reachable sets still have to agree, and the wider padding must stay inert.
    small_metadata = _metadata([0, 16, 32])
    idx_q2 = torch.randn(1, 32, hi, di, requires_grad=True)
    idx_k2 = torch.randn(1, 16, di, requires_grad=True)
    weights2 = torch.randn(1, 32, hi, requires_grad=True)
    ei2, es2, _ = eager_indexer.forward(idx_q2, idx_k2, weights2, small_metadata, candidates_BL1C=None)
    fi2, fs2, _ = fused_indexer.forward(idx_q2, idx_k2, weights2, small_metadata, candidates_BL1C=None)
    assert fi2.shape[-1] == fused_indexer.index_topk == 32
    assert ei2.shape[-1] == 16  # the eager selector clamps to the pool
    assert fs2.dtype == torch.float32
    # Same cross-frame shift as above: the fused selection is document-local.
    starts2 = small_metadata.kernel.frame_for(eager_indexer.compress_ratio).cu_seqlens.to(torch.long)
    docs2 = small_metadata.ref.doc_ids_BL.reshape(-1)
    for t in range(32):
        fused_t = {int(starts2[docs2[t]]) + int(j) for j in fi2[0, t].tolist() if j >= 0}
        eager_t = {int(j) for j in ei2[0, t].tolist() if j >= 0}
        assert fused_t == eager_t, t


def test_the_teacher_reaching_slikg_is_normalised_by_seqlen(monkeypatch):
    """The indexer's objective is a mean over tokens, and the loss cannot apply it.

    The teacher is injected straight into SLIKG on ``topk_scores``' gradient, so the
    trainer's ``global_valid_tokens`` division never touches it; the port divides by the
    row's token count itself -- the query's TND leading axis, i.e. the packed ``seqlen``.

    One index source owns one selection and drives one SLIKG call, and the teacher that
    layer writes is its own ``p`` divided by the token count.  Which sources those are comes
    from the layer table rather than from the run, and the run is checked against it; the
    expected value is then named from that count and the token count, not read back out of
    the same run.  The constant is made larger than the token count, so a teacher that never
    got divided lands far outside what the tables allow.

    The ``-1`` padding slots are the exception, and must be read around here: SMLAG leaves
    them non-zero (the kernel computes a marginal for a slot the selection never reached),
    so the port zeroes them on the way out -- SLIKG's ``ReduceSumVf`` sums every slot, and
    only both sides agreeing on the zero keeps a padded slot out of ``dI = Z * Y - p``.
    A padded slot therefore says nothing about the division, and the slots that do are the
    unpadded ones.
    """
    seen = {"calls": 0, "teacher": [], "padding": [], "tokens": []}
    constant = 1024.0

    def fake_slig(q, k, w, sparse_indices, attn_softmax_l1_norm, **kw):
        seen["teacher"].append(attn_softmax_l1_norm.detach().clone())
        seen["padding"].append(sparse_indices < 0)
        seen["tokens"].append(q.shape[0])
        seen["calls"] += 1
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(w).float(),
            torch.zeros_like(attn_softmax_l1_norm),
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", fake_slig)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kw: (q.clone(), q.new_zeros(1, q.shape[0], q.shape[1])),
    )

    def fake_smlag(q, grad, *args, **kw):
        indices = kw["cmp_sparse_indices"]
        teacher = torch.full_like(indices, constant, dtype=torch.float32) if indices is not None else None
        shared = kw["cmp_kv"]
        return (
            grad,
            torch.zeros_like(kw["ori_kv"]),
            torch.zeros_like(shared) if shared is not None else None,
            torch.zeros_like(kw["sinks"]),
            None,
            teacher,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", fake_smlag)
    model = _tiny_model(monkeypatch, fused=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run_forward_backward(model)

    assert seen["calls"], "SLIKG must be reached for the indexer to train"
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    # One SLIKG call per index source: the override trains each source's selection, and the
    # debug flavor shares the flash topology's tables, so the count comes from there and not
    # from a number read back out of this run.
    sources = _teacher_sources(registry.V41_FULL_INDEX_SOURCE_LAYERS)
    assert seen["calls"] == len(sources), (seen["calls"], sources)
    assert len(set(seen["tokens"])) == 1, set(seen["tokens"])

    for teacher, padding in zip(seen["teacher"], seen["padding"], strict=True):
        tokens = seen["tokens"][0]
        assert tokens == _TOKENS.shape[1], (tokens, _TOKENS.shape[1])
        assert constant > tokens, "the expected value below only bites while the divider is smaller"
        # The ``-1`` slots are SMLAG's junk zeroed out on the way here; the unpadded ones
        # are what the division acts on.
        torch.testing.assert_close(
            teacher.masked_select(padding),
            torch.zeros(int(padding.sum())),
            rtol=0,
            atol=0,
        )
        value = teacher.masked_select(~padding).max().item()
        # SLIKG's teacher is ``constant`` scaled by the layer's own weight, which cancels:
        # what is left is an integer multiple of ``constant / seqlen``, here 1024/128 = 8,
        # so the multiplier is 4 or 6.  An undivided teacher would be 512 times larger,
        # well outside anything the tables allow.  Note what this does *not* catch: the
        # frame has a single length, so a divisor that was uniformly wrong by an integer
        # factor would still be an integer multiple and still fit the bound.  Only a second
        # packed length could tell ``seqlen`` from a wrong factor of it.
        contributors = value * tokens / constant
        assert abs(contributors - round(contributors)) < 1e-6, (
            f"the teacher is not an integer multiple of constant/seqlen ({constant}/{tokens}): got {value}"
        )
        assert 1 <= round(contributors) <= seen["calls"], (
            f"the teacher implies {round(contributors)} contributors but SLIKG ran "
            f"{seen['calls']} times, so the seqlen division ({tokens}) is missing"
        )
        expected = torch.full_like(teacher, constant * round(contributors) / tokens).masked_fill(padding, 0.0)
        torch.testing.assert_close(teacher, expected)


def test_the_teacher_carrier_is_on_the_operands_device(monkeypatch):
    """The carrier must not land on the default device.

    A Function's edges are its Tensor arguments, and the engine checks each returned
    gradient's device against its input's.  The teacher is emitted by
    ``sparse_flash_mla_grad`` wherever its operands live, so a carrier built on the
    default device fails backward on the NPU with

        Function _SparseMLABackward returned an invalid gradient at index 5 -
        expected device cpu but got npu:0

    (index 5 = the carrier, once the non-Tensor arguments are dropped from the edge
    list).  ``meta`` stands in for "not the default device" here, since the CPU harness
    has no second device to hand.
    """
    from torchtitan_npu.override.deepseek_v4_1.lightning_indexer import ascendc as li

    device = torch.device("meta")
    seen = {}

    def fake_kernel(q, k, w, descale_q, descale_k, topk, quant_mode, **kwargs):
        total = q.shape[0]
        indices = torch.zeros(total, 1, topk, dtype=torch.int32, device=device)
        # The carrier is an output of the Function either way, so the operator's own
        # second output is the empty-values slot a ``return_value=False`` call produces.
        values = torch.zeros(0, dtype=torch.bfloat16, device=device)
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return indices, values, empty, empty

    def fake_metadata(*args, **kwargs):
        return torch.zeros(8, dtype=torch.int32, device=device)

    monkeypatch.setattr(torch.ops.cann_ops_transformer.ds41, "quant_lightning_indexer", fake_kernel)
    monkeypatch.setattr(li, "_kernel_options", lambda *a, **k: {})
    monkeypatch.setattr(li, "_kernel_geometry", lambda *a, **k: {})
    monkeypatch.setattr(li, "_pack_mxfp4", _fake_pack_mxfp4)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer.ds41, "quant_lightning_indexer_metadata", fake_metadata
    )

    original = li._LightningIndexerTND.forward

    def capture(ctx, *args):
        out = original(ctx, *args)
        seen["indices"] = out[0].device
        seen["carrier"] = out[1].device
        return out

    monkeypatch.setattr(li._LightningIndexerTND, "forward", staticmethod(capture))

    selector = _tiny_model(monkeypatch, fused=True).layers["2"].attention.indexer.selector
    heads, head_dim = selector.num_index_heads, selector.index_head_dim
    selector.forward(
        torch.zeros(1, 8, heads, head_dim, dtype=torch.bfloat16, device=device),
        torch.zeros(1, 4, head_dim, dtype=torch.bfloat16, device=device),
        torch.zeros(1, 8, heads, dtype=torch.float32, device=device),
        _metadata([0, 8], compress_ratios=(1,)),
        candidates_BL1C=None,
    )

    assert seen["indices"] == device, seen["indices"]
    assert seen["carrier"] == device, (
        f"the carrier must share the selection's device, got {seen['carrier']} "
        f"against {seen['indices']}"
    )


def test_both_kernels_receive_the_layout_their_contract_requires(monkeypatch):
    """The two boundaries where layout has to be translated, and the one where it must not.

    ``layout_q="TND"`` means a ``[T, N2, K]`` selection at SMLA, while the model carries it
    as ``[B, L, 1, K]``.  SLIKG is stricter: its tiling requires ``attn_softmax_l1_norm`` to
    match ``sparse_indices`` exactly.  Both are silent in the CPU fakes -- they accept any
    shape -- so this pins them here, since a wrong rank only surfaces on device.
    """
    from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc as sa

    seen = {}
    real_apply = sa._SparseMLA.apply

    def spy_apply(q, swa_k, cmp_k, topk_indices, sinks, masks, scale, ratio, window, tscore, wants):
        seen["smla_indices"] = tuple(topk_indices.shape)
        seen["smla_carrier"] = None if tscore is None else tuple(tscore.shape)
        return real_apply(q, swa_k, cmp_k, topk_indices, sinks, masks, scale, ratio, window, tscore, wants)

    def fake_slig(q, k, w, sparse_indices, attn_softmax_l1_norm, **kw):
        seen["slig_indices"] = tuple(sparse_indices.shape)
        seen["slig_teacher"] = tuple(attn_softmax_l1_norm.shape)
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(w).float(),
            torch.zeros_like(attn_softmax_l1_norm),
        )

    monkeypatch.setattr(sa._SparseMLA, "apply", staticmethod(spy_apply))
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", fake_slig)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kw: (q.clone(), q.new_zeros(1, q.shape[0], q.shape[1])),
    )

    def fake_smlag(q, grad, *args, **kw):
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

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", fake_smlag)
    model = _tiny_model(monkeypatch, fused=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run_forward_backward(model)

    seq_len = _TOKENS.shape[1]
    topk = model.layers["2"].attention.indexer.index_topk
    # Model-facing ``[B, L, 1, K]`` becomes the kernel's ``[T, N2, K]`` at both SMLA inputs.
    assert seen["smla_indices"] == seen["smla_carrier"] == (seq_len, 1, topk)
    # And SLIKG gets the teacher in exactly the selection's shape -- that equality is what
    # its tiling checks.
    assert seen["slig_indices"] == seen["slig_teacher"] == (seq_len, 1, topk)


def test_the_selection_is_sorted_with_padding_last(monkeypatch):
    """The model contract is one position order with the padding at the tail.

    The kernel's own order is unspecified, so the port re-sorts.  The direction is ours --
    neither the kernel schema nor the operator docs fix one, and the reference happens to
    sort ascending -- and is descending, because keying the sort on ``-index`` pushes
    ``-1`` behind every valid entry for free.

    The fake is made to emit a deliberately unsorted order, because a fake that
    pre-sorts cannot see whether the port sorts at all.

    Both the indices and the scores must move together: the scores are the carrier whose
    gradient is the teacher, and the teacher is per slot.
    """
    real_fake = _fake_qli

    def unsorted_fake(*args, **kw):
        indices, values, block_indices, block_length = real_fake(*args, **kw)
        return (
            indices.flip(-1).contiguous(),
            values.flip(-1).contiguous(),
            block_indices,
            block_length,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer.ds41, "quant_lightning_indexer", unsorted_fake)
    model = _tiny_model(monkeypatch, fused=True)
    _, _, kwargs = model.build_attention_masks(_TOKENS, _TOKENS, {"positions": _POSITIONS})
    metadata = kwargs["attention_masks"]

    selector = model.layers["2"].attention.indexer.selector
    # requires_grad on the operands, as in a real layer: an autograd Function only gives
    # its outputs a grad_fn when one of its inputs needs a gradient, and the carrier's edge
    # is the whole point of the second output.
    q = torch.randn(1, 128, selector.num_index_heads, selector.index_head_dim, requires_grad=True)
    k = torch.randn(1, 64, selector.index_head_dim, requires_grad=True)
    w = torch.randn(1, 128, selector.num_index_heads, requires_grad=True)
    indices, scores, _ = selector.forward(q, k, w, metadata, candidates_BL1C=None)

    checked = 0
    for t in range(indices.shape[1]):
        row = indices[0, t]
        valid = row[row >= 0]
        if valid.numel() < 2:
            continue
        # Descending positions, and no valid entry after the first padding.
        assert torch.all(valid[1:] < valid[:-1]), (t, valid[:8].tolist())
        first_pad = int((row < 0).nonzero()[0]) if bool((row < 0).any()) else row.numel()
        assert torch.all(row[first_pad:] < 0), t
        checked += 1
    assert checked > 0, "the batch must contain rows with a real selection"
    # The carrier is a fabricated buffer of the selection's shape, produced by the
    # autograd Function so that the teacher's gradient has a path back to the indexer.
    # Its values are read by neither side, so nothing is asserted about them.
    assert scores.shape == indices.shape
    assert scores.requires_grad


def test_lightning_indexer_asc_stack_applies_without_claim_conflicts(monkeypatch):
    """The fused node override composes with the blanket norm/rope stacks.

    ``lightning_indexer.asc`` swaps only the parameterless ``Selector`` node, which
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
                "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
                "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
            ]
        ),
        model,
    )

    indexer = model.layers[2].attention.indexer
    assert isinstance(indexer.selector, AscSelector.Config)
    assert isinstance(indexer.rope, ComplexRoPE.Config)
    assert isinstance(model.layers[0].attention.q_norm, AscRMSNorm.Config)
    assert isinstance(model.layers[0].attention_norm, AscRMSNorm.Config)
    assert isinstance(model.layers[0].attention.rope, AscComplexRoPE.Config)


def test_full_model_smla_trains_the_indexer_through_the_teacher_edge(monkeypatch):
    from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc

    model = _tiny_model(monkeypatch, fused=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
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
    # The indexer is trained only through the teacher edge, so a finite gradient on
    # every indexer parameter is what proves the teacher reached SLIKG -- on every
    # layer, the pool group included.
    indexer_params = [(n, p) for n, p in model.named_parameters() if ".indexer." in n]
    assert indexer_params, "the model must own indexer parameters"
    for name, parameter in indexer_params:
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_a_half_fused_teacher_pair_is_rejected(monkeypatch):
    """One fused side without the other is refused at build time.

    Both halves are compile-time facts on the two configs, so a half-fused stack is a
    configuration error rather than something to discover from a flat training curve.
    """
    from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
    from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention

    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")

    def config(*, fused_selector: bool, fused_core: bool):
        cfg = registry.model_registry("deepseek_v4_1_debugmodel").model
        for layer in cfg.layers:
            # The pair only exists where there is a compressed stream.
            if layer.attention.inner_attention.compress_ratio == 0:
                continue
            if fused_selector:
                layer.attention.indexer.selector = derive(layer.attention.indexer.selector, AscSelector.Config)
            if fused_core:
                layer.attention.inner_attention = derive(
                    layer.attention.inner_attention, AscV41SparseAttention.Config
                )
        return cfg

    assert not CompressedSparseInnerAttention2.Config.provides_indexer_teacher

    # Selector only: topk_scores would carry no teacher at all.
    with (
        pytest.raises(ValueError, match=r"lightning_indexer.asc is active without sparse_attn.asc"),
        torch.random.fork_rng(devices=[]),
    ):
        build_cpu_model(config(fused_selector=True, fused_core=False))

    # Core only: the teacher edge would have no consumer.
    with (
        pytest.raises(ValueError, match=r"sparse_attn.asc is active without lightning_indexer.asc"),
        torch.random.fork_rng(devices=[]),
    ):
        build_cpu_model(config(fused_selector=False, fused_core=True))

    # Both together: the pair is complete, so the model builds.
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config(fused_selector=True, fused_core=True))
    assert isinstance(model.layers["2"].attention.inner_attention, AscV41SparseAttention)
    assert isinstance(model.layers["2"].attention.indexer.selector, AscSelector)


@pytest.mark.parametrize(
    "imports",
    [
        ["torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc"],
        ["torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc"],
    ],
    ids=["selector-only", "core-only"],
)
def test_a_half_fused_teacher_pair_is_rejected_through_override_imports(monkeypatch, imports):
    """The rejection fires on the real ``override.imports`` wiring, not just on hand-built configs."""
    from torchtitan.config.override import OverrideConfig, apply_overrides

    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    config = registry.model_registry("deepseek_v4_1_debugmodel").model
    apply_overrides(OverrideConfig(imports=imports), config)

    with (
        pytest.raises(ValueError, match="must be enabled together"),
        torch.random.fork_rng(devices=[]),
    ):
        build_cpu_model(config)


def test_the_teacher_sources_are_the_index_source_layers():
    """The count the seqlen test leans on, pinned against the shipped table.

    It is the whole reason that test can name an expected teacher value instead of reading
    one back out of the run, so it is checked here rather than only inside a forward.
    """
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    sources = _teacher_sources(registry.V41_FULL_INDEX_SOURCE_LAYERS)
    assert sources == (2, 8, 14, 20, 24, 28, 32, 36)
    # The debug flavor the seqlen test builds shares this topology, so the frame it measures
    # has this many teacher-carrying layers.
    assert len(sources) == 8


def test_the_layer_role_picks_the_kernel(monkeypatch):
    """``mode`` plus the capacity decide which operator each layer calls.

    There is no separate source flag: Full Mode is the role that owns the compressed KV the
    pool indexes, so it is the only one that can build a pool, and a Reindex Mode layer
    carrying a capacity is therefore necessarily a searcher.  The roles are read off the
    shipped layer table and checked against the calls the model actually makes, rather than
    against a hand-built config that could agree with a wrong rule.

    The pool itself is checked for shape, and every searcher is checked to be reading the
    same capacity the one producer was configured with.
    """
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    searchers = {
        str(layer)
        for layer in registry.V41_FULL_INDEX_SOURCE_LAYERS
        if layer > registry.V41_CANDIDATE_SOURCE_LAYER
    }
    pool_free = {
        str(layer) for layer in registry.V41_FULL_INDEX_SOURCE_LAYERS if layer < registry.V41_CANDIDATE_SOURCE_LAYER
    }

    seen = {"qli": [], "qsli": []}

    def recording_qli(*args, **kwargs):
        seen["qli"].append(kwargs.get("candidate_topk_blocks", -1))
        return _fake_qli(*args, **kwargs)

    def recording_qsli(*args, **kwargs):
        seen["qsli"].append(args[4].shape)
        return _fake_qsli(*args, **kwargs)

    monkeypatch.setattr(torch.ops.cann_ops_transformer.ds41, "quant_lightning_indexer", recording_qli)
    monkeypatch.setattr(torch.ops.cann_ops_transformer.ds41, "quant_sparse_lightning_indexer", recording_qsli)

    model = _tiny_model(monkeypatch, fused=True)
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kw: (q.clone(), q.new_zeros(1, q.shape[0], q.shape[1])),
    )

    def smlag_backward(q, grad, *args, **kw):
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

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", smlag_backward)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run_forward_backward(model)

    # One call per index source, split by role: the searchers all run the consumer, and the
    # source is the only layer that asks the producer for a pool.
    #
    # The capacities here are the debug flavor's, and they are *not* kernel-legal -- it runs
    # a 32-wide index head and a capacity of 4, where the operators fix 128 and 2048.  The
    # stand-ins accept them, so this test pins the dispatch and the wiring, not the
    # geometry; the shipped flash flavor (2048/8, head 128) is what the kernels take.
    assert len(seen["qsli"]) == len(searchers), (len(seen["qsli"]), searchers)
    assert len(seen["qli"]) == len(pool_free) + 1, (len(seen["qli"]), pool_free)
    pooled = [capacity for capacity in seen["qli"] if capacity > 0]
    assert len(pooled) == 1, f"exactly one layer builds the pool, got {pooled}"
    assert pooled[0] > 0
    # The consumer is handed the kernel's own ``[T, 1, capacity]``, and every searcher reads
    # the same capacity the producer was configured with.  The model-facing
    # ``[B, L, 1, capacity]`` never reaches the operator: the Function owns that translation.
    #
    # The rank is pinned, not just the capacity: the table passes through every searcher
    # unsqueezed, so a translation applied at the caller instead of on the kernel edge would
    # add a batch axis per layer and still satisfy a capacity-only assertion.
    for shape in seen["qsli"]:
        assert len(shape) == 3 and shape[1] == 1, shape
        assert shape[2] == pooled[0], (shape, pooled[0])

def test_the_override_entry_carries_the_legacy_switch(monkeypatch):
    """The entry's JSON argument is what sets the switch, not a hand-set field.

    The override matches ``Selector.Config`` exactly, so it can only be applied to a config
    that has not been derived yet -- which is also the only order a run uses.  Checking the
    entry here, and the field's effect separately, keeps a wrong entry signature from hiding
    behind a test that set the field itself.

    The entry text goes through the CLI parser rather than being assembled as a
    ``(target, kwargs)`` pair, because that quoted ``target={...}`` string is exactly what
    the launch script passes -- a syntax error there would not show up in a hand-built pair.
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides, parse_cli_imports

    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")

    def selectors(*tokens):
        config = registry.model_registry("deepseek_v4_1_debugmodel").model
        apply_overrides(OverrideConfig(imports=parse_cli_imports(list(tokens))), config)
        return [cfg for _, cfg, _, _ in config.traverse(Selector.Config)]

    plain = selectors("torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc")
    assert plain and all(isinstance(cfg, AscSelector.Config) for cfg in plain)
    assert all(cfg.legacy is True for cfg in plain), "the switch defaults to the pool-free kernel"

    switched = selectors(
        'torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc={"legacy":false}'
    )
    assert switched and all(isinstance(cfg, AscSelector.Config) for cfg in switched)
    assert all(cfg.legacy is False for cfg in switched)
    # The switch is additive: the contract the teacher pair is checked on is untouched.
    assert all(cfg.consumes_indexer_teacher is True for cfg in switched)


def test_the_legacy_switch_routes_every_source_to_the_unquantized_kernel(monkeypatch):
    """With the switch on, no index source reaches the quantized pair.

    The legacy operator has no pool, so this is also the pool-free baseline the pooled path
    is measured against: every index source selects on its own, and the quantized operators
    are never called at all.
    """
    seen = {"legacy": 0, "qli": 0, "qsli": 0}

    def recording_legacy(*args, **kwargs):
        seen["legacy"] += 1
        return _fake_legacy_li(*args, **kwargs)

    def recording_qli(*args, **kwargs):
        seen["qli"] += 1
        return _fake_qli(*args, **kwargs)

    def recording_qsli(*args, **kwargs):
        seen["qsli"] += 1
        return _fake_qsli(*args, **kwargs)

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer", recording_legacy)
    monkeypatch.setattr(torch.ops.cann_ops_transformer.ds41, "quant_lightning_indexer", recording_qli)
    monkeypatch.setattr(torch.ops.cann_ops_transformer.ds41, "quant_sparse_lightning_indexer", recording_qsli)

    model = _tiny_model(monkeypatch, fused=True, legacy=True)
    selector = model.layers["2"].attention.indexer.selector
    assert selector.legacy is True
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fake_li_metadata)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _fake_slig_metadata)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kw: (q.clone(), q.new_zeros(1, q.shape[0], q.shape[1])),
    )

    def smlag_backward(q, grad, *args, **kw):
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

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", smlag_backward)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _run_forward_backward(model)

    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    sources = _teacher_sources(registry.V41_FULL_INDEX_SOURCE_LAYERS)
    assert seen["legacy"] == len(sources), (seen, sources)
    assert seen["qli"] == 0 and seen["qsli"] == 0, seen
