import pytest
import torch

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from tests.unit_tests.override.deepseek_v4_1.test_sparse_fused_indices import _metadata
from torchtitan_npu.models.deepseek_v4_1.indexer import IndexerDistillLoss
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc


@pytest.mark.parametrize("mode", ["logits", "teacher"])
@pytest.mark.parametrize(
    "checkpointed",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not torch.npu.is_available(),
                reason=(
                    "checkpoint recompute enters NPU autocast, which lazy-initializes "
                    "the NPU runtime; CPU-only CI runners have no device"
                ),
            ),
        ),
    ],
)
def test_smla_reuses_teacher_without_loss_forward(monkeypatch, mode, checkpointed):
    meta = torch.zeros(1, dtype=torch.int32)
    monkeypatch.setattr(ascendc, "sparse_flash_mla_metadata", lambda *a, **k: meta)
    monkeypatch.setattr(ascendc, "sparse_flash_mla_grad_metadata", lambda *a, **k: meta)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **k: (q.clone(), torch.zeros(1, q.shape[0], q.shape[1])),
    )

    def backward(q, *args, **kwargs):
        p = torch.tensor([[[0.1, 0.3, 9.0]]]).expand(q.shape[0], 1, 3)
        return (
            torch.zeros_like(q),
            torch.zeros_like(kwargs["ori_kv"]),
            torch.zeros_like(kwargs["cmp_kv"]),
            torch.zeros_like(kwargs["sinks"]),
            None,
            p,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", backward)

    def forbidden(*args, **kwargs):
        pytest.fail("_teacher must not run under SMLA")

    monkeypatch.setattr(IndexerDistillLoss, "_teacher", forbidden)
    attn = build_cpu_model(
        ascendc.AscV41SparseAttention.Config(
            window_size=2,
            softmax_scale=0.5,
            compress_ratio=1,
            score_gradient=mode,
            aux_loss=IndexerDistillLoss.Config(coeff=2.0, reduce_mesh="batch", global_batch_size=1, softmax_scale=0.5),
        )
    )
    metadata = _metadata(torch.arange(4))
    scores = torch.tensor([[[-float("inf"), 1.0, 0.0]] * 4], requires_grad=True)
    indices = torch.tensor([[[-1, 0, 1]] * 4])
    indices[:, 3] = -1
    q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16, requires_grad=True)

    def consume(q, scores):
        return attn(
            q,
            q.squeeze(2),
            q.squeeze(2),
            attention_masks=metadata,
            topk_indices=indices,
            topk_scores=scores,
            attn_sink=torch.zeros(1),
        )

    if checkpointed:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(consume, q, scores, use_reentrant=False)
    else:
        out = consume(q, scores)
    # A second consumer must add its own contribution to the same scores.
    out2 = consume(q, scores)
    ((out.sum() + out2.sum()) * 0).backward()
    # What this test owns is the plumbing, not the kernel's numbers: the fake
    # backward reports a mass over all three slots while the pure-Python loss
    # renormalises over the reachable ones, so the magnitudes legitimately differ.
    # The fused path must distil exactly the reachable slots, drop a row with no
    # reachable entry, and let both consumers accumulate onto the same logits.
    valid = (indices >= 0).reshape(4, 3)
    grad = scores.grad.reshape(4, 3)
    # Slots the selection never reaches take no gradient.
    assert torch.equal(grad[~valid], torch.zeros(int((~valid).sum())))
    # Row 3 has no reachable entry at all: it trains nothing.
    assert not valid[3].any()
    assert torch.equal(grad[3], torch.zeros(3))
    # Both consumers contributed, and the reachable slots carry the gradient.
    assert (grad[:3][valid[:3]] != 0).all()
    assert torch.isfinite(attn.aux_loss.read())
    assert float(attn.aux_loss.read()) > 0.0


@pytest.mark.parametrize("ratio,training,aux", [(0, True, True), (1, False, True), (2, True, False)])
def test_attention_without_distillation_has_no_scores_edge(monkeypatch, ratio, training, aux):
    captured = []

    def kernel(*args):
        captured.append(args)
        return args[0].clone(), torch.zeros(1, 4, 1)

    monkeypatch.setattr(ascendc._SparseMLA, "apply", kernel)
    loss = (
        IndexerDistillLoss.Config(coeff=1.0, reduce_mesh="batch", global_batch_size=1, softmax_scale=0.5)
        if aux
        else None
    )
    attention = build_cpu_model(
        ascendc.AscV41SparseAttention.Config(window_size=2, softmax_scale=0.5, compress_ratio=ratio, aux_loss=loss)
    )
    attention.train(training)
    q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16)
    indices = torch.zeros(1, 4, 1, dtype=torch.long) if ratio else None
    shared = torch.zeros(1, 4 // ratio, 4, dtype=torch.bfloat16) if ratio else None
    out = attention(
        q,
        q.squeeze(2),
        shared,
        attention_masks=_metadata(torch.arange(4)),
        topk_indices=indices,
        topk_scores=torch.ones(1, 4, 1, requires_grad=True),
        attn_sink=torch.zeros(1),
    )
    torch.testing.assert_close(out, q)
    assert captured[0][11] is None
    assert captured[0][12] is None
