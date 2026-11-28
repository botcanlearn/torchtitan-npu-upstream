# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.observability.sdc_replayer import SDCReplayer, SDCReplayMismatch

from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable


class _ToyEngramModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = HostEngramTable.Config(
            vocab_size=16,
            layer_id=1,
            ngram_orders=(2,),
            num_heads=1,
            head_vocab_sizes=(16,),
            embedding_dim=4,
            num_embeddings=16,
            require_token_id_map=False,
        ).build()
        self.proj = torch.nn.Linear(4, 2, bias=False)


def _run_ga2(model: _ToyEngramModel, rows: list[torch.Tensor], grads: list[torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros((), dtype=torch.float32)
    for row_ids, grad_rows in zip(rows, grads, strict=True):
        model.table.accumulate_sparse_gradient(row_ids, grad_rows)
        out = model.proj(grad_rows).sum()
        out.backward()
        loss = loss + out.detach()
    return loss


@pytest.mark.parametrize("num_replays", [1, 2])
def test_sdc_replayer_restores_host_engram_sparse_grad(num_replays: int) -> None:
    torch.manual_seed(0)
    ref_model, replay_model = _ToyEngramModel(), _ToyEngramModel()
    replay_model.load_state_dict(ref_model.state_dict())

    rows = [torch.tensor([1, 3, 1], dtype=torch.int64), torch.tensor([3, 7], dtype=torch.int64)]
    grads = [
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5], [0.25, 0.5, 0.75, 1.0]]),
        torch.tensor([[2.0, 1.0, 0.5, 0.25], [3.0, 2.0, 1.0, 0.5]]),
    ]
    ref_loss = _run_ga2(ref_model, rows, grads)
    ref_sparse = ref_model.table.pending_sparse_grad()
    assert ref_sparse is not None

    replayer = SDCReplayer.Config(num_steps=1, num_replays=num_replays).build(
        modules=[replay_model],
        device=torch.device("cpu"),
    )

    calls = 0

    def execute() -> torch.Tensor:
        nonlocal calls
        calls += 1
        return _run_ga2(replay_model, rows, grads)

    loss = replayer.run_fwd_bwd(execute, step=1, get_loss=lambda v: v)
    replayed_sparse = replay_model.table.pending_sparse_grad()
    assert calls == 1 + num_replays
    assert torch.equal(loss, ref_loss)
    assert replayed_sparse is not None
    assert "table._pending_sparse_grad" not in replay_model.state_dict()
    assert torch.equal(replayed_sparse.indices(), ref_sparse.indices())
    assert torch.equal(replayed_sparse.values(), ref_sparse.values())

    replay_model.bfloat16()
    after_apply = replay_model.table.pending_sparse_grad()
    assert after_apply is not None
    assert after_apply.device.type == "cpu"
    assert after_apply.dtype == torch.float32
    assert torch.equal(after_apply.values(), ref_sparse.values())


def test_sdc_replayer_detects_corrupted_host_engram_sparse_grad() -> None:
    torch.manual_seed(0)
    model = _ToyEngramModel()
    rows = [torch.tensor([2, 5], dtype=torch.int64)]
    grads = [torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])]

    replayer = SDCReplayer.Config(num_steps=1, num_replays=1).build(
        modules=[model],
        device=torch.device("cpu"),
    )

    calls = 0

    def execute() -> torch.Tensor:
        nonlocal calls
        calls += 1
        # Swap the two rows' sparse gradient values on replay while keeping loss
        # and dense grads bit-identical; order-insensitive XOR would miss this.
        sparse_grad = grads[0].flip(0) if calls == 2 else grads[0]
        model.table.accumulate_sparse_gradient(rows[0], sparse_grad)
        out = model.proj(grads[0]).sum()
        out.backward()
        return out.detach()

    with pytest.raises(SDCReplayMismatch, match=r"buffer:0:table\._pending_sparse_grad"):
        replayer.run_fwd_bwd(execute, step=1, get_loss=lambda v: v)
