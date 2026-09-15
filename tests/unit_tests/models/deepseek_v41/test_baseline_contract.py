# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.models.common.attention import VarlenMetadata

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v41.config_registry import deepseek_v41_debugmodel
from torchtitan_npu.models.deepseek_v41.metadata import build_compressed_varlen_metadata
from torchtitan_npu.models.deepseek_v41.model_registry import _make_v41_moe_config
from torchtitan_npu.models.deepseek_v41.reference import ReferenceMetadataExtension


def test_plain_weights_use_routed_experts_and_track_empty_experts():
    config = _make_v41_moe_config(
        layer_id=0,
        dim=8,
        moe_inter_dim=8,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=1e-3,
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
    )
    with torch.random.fork_rng(devices=[]):
        moe = build_cpu_model(config)
    with torch.no_grad():
        moe.router.gate.weight.zero_()
        moe.expert_bias_E.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    calls = []
    handle = moe.routed_experts.register_forward_hook(lambda module, args, output: calls.append(output))
    try:
        x = torch.linspace(-1, 1, 24).reshape(1, 3, 8).requires_grad_()
        out = moe(x)
        assert len(calls) == 1
        torch.testing.assert_close(moe.tokens_per_expert_E, torch.tensor([3.0, 3.0, 0.0, 0.0]))
        out.square().mean().backward()
        for parameter in (
            moe.routed_experts.inner_experts.w1_EFD,
            moe.routed_experts.inner_experts.w2_EDF,
            moe.routed_experts.inner_experts.w3_EFD,
        ):
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad[2:]) == 0
        assert x.grad is not None and torch.isfinite(x.grad).all()
    finally:
        handle.remove()


def test_baseline_rejects_tensor_parallel(monkeypatch):
    monkeypatch.setenv("USE_GOLDEN", "1")
    config = deepseek_v41_debugmodel()
    config.parallelism.tensor_parallel_degree = 2
    with pytest.raises(NotImplementedError, match="TP=1"):
        config.model_spec.model.update_from_config(config=config)


def test_reference_masks_preserve_short_document_tails():
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    varlen = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=3, max_k=3)
    common = build_compressed_varlen_metadata(varlen, (0, 1, 2))
    metadata = ReferenceMetadataExtension(ReferenceMetadataExtension.Config(window_size=2, materialized_ratios=(1,)))(
        common
    )
    expected = torch.tensor([[False, False], [True, False], [False, False], [False, True], [False, True]])
    torch.testing.assert_close(metadata.reference.ratios[2].dense_mask[0, 0], expected)
    assert metadata.reference.ratios[1].dense_mask.shape == (1, 1, 5, 5)
