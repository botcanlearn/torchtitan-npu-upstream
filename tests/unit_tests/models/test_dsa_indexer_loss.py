# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verify DSAIndexerLoss is compatible with TP parallelize_module."""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DSAIndexerLoss is a torchtitan Module (inherits from torchtitan.protocols.module.Module)
# ---------------------------------------------------------------------------


def test_dsa_indexer_loss_is_torchtitan_module():
    """DSAIndexerLoss inherits from torchtitan Module."""
    from torchtitan.protocols.module import Module as TorchTitanModule

    from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLoss

    assert issubclass(DSAIndexerLoss, nn.Module)
    assert issubclass(DSAIndexerLoss, TorchTitanModule)


# ---------------------------------------------------------------------------
# Submodule traversal: DSAIndexerLoss can be found via get_submodule
# ---------------------------------------------------------------------------


class MockInnerAttention(nn.Module):
    def __init__(self):
        super().__init__()
        from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLoss

        self.compute_dsa_indexer_loss = DSAIndexerLoss.Config().build()


class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_attention = MockInnerAttention()


class MockTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MockAttention()


def test_dsa_indexer_loss_findable_via_get_submodule():
    """parallelize_module uses get_submodule to resolve path-based plans."""
    block = MockTransformerBlock()

    sub = block.get_submodule("attention.inner_attention.compute_dsa_indexer_loss")
    from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLoss

    assert isinstance(sub, DSAIndexerLoss)


def test_dsa_indexer_loss_findable_via_named_children():
    """parallelize_module uses named_children + fnmatch to walk the plan dict."""
    block = MockTransformerBlock()

    inner = block.attention.inner_attention
    from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLoss

    found = [(n, m) for n, m in inner.named_children() if isinstance(m, DSAIndexerLoss)]
    assert len(found) == 1
    assert found[0][0] == "compute_dsa_indexer_loss"


# ---------------------------------------------------------------------------
# PrepareModuleInput compatibility: forward signature is inspectable
# ---------------------------------------------------------------------------


def test_dsa_indexer_loss_forward_signature_inspectable():
    """PrepareModuleInput._apply uses register_forward_pre_hook on nn.Module."""
    from torchtitan_npu.models.common.dsa_indexer_loss import DSAIndexerLoss

    mod = DSAIndexerLoss.Config().build()

    # register_forward_pre_hook works on any nn.Module
    hook_called = []

    def hook(module, inputs):
        hook_called.append(inputs)
        return inputs

    mod.register_forward_pre_hook(hook)

    x = torch.randn(2, 4)
    y = torch.randn(2, 4)
    topk = torch.randint(0, 4, (2, 4))
    loss = mod(x, y, topk, 1.0)

    assert len(hook_called) == 1
    assert loss.dim() == 0


# ---------------------------------------------------------------------------
# parallelize_module path-based plan: can target DSAIndexerLoss
# ---------------------------------------------------------------------------


def test_parallelize_module_targets_torchtitan_module():
    """parallelize_module with dict plan finds and wraps torchtitan Module submodules."""
    block = MockTransformerBlock()

    inner = block.get_submodule("attention.inner_attention.compute_dsa_indexer_loss")
    assert isinstance(inner, nn.Module)

    # Simulate what PrepareModuleInput does: wrapping forward
    original_forward = inner.forward

    def wrapped_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)

    inner.forward = wrapped_forward

    x = torch.randn(2, 4)
    y = torch.randn(2, 4)
    topk = torch.randint(0, 4, (2, 4))
    loss = inner(x, y, topk, 1.0)
    assert loss.dim() == 0


# ---------------------------------------------------------------------------
# AutoScaler / LoggingHelper: shared class works with both model paths
# ---------------------------------------------------------------------------


def test_auto_scaler_set_loss_scale():
    from torchtitan_npu.models.common import dsa_indexer_loss

    scale = torch.tensor(0.5)
    dsa_indexer_loss.DSAIndexerLossAutoScaler.set_loss_scale(scale)
    assert dsa_indexer_loss.LOSS_SCALE is not None
    assert dsa_indexer_loss.LOSS_SCALE.item() == 0.5


def test_logging_helper_save_and_clean():
    from torchtitan_npu.models.common.dsa_indexer_loss import (
        DSAIndexerLossLoggingHelper,
    )

    DSAIndexerLossLoggingHelper.tracker.clear()
    loss = torch.tensor(0.123)
    DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=4)
    assert "values" in DSAIndexerLossLoggingHelper.tracker
    assert DSAIndexerLossLoggingHelper.tracker["values"][0].item() == pytest.approx(
        0.123
    )

    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    assert DSAIndexerLossLoggingHelper.tracker["values"][0].item() == pytest.approx(0.0)


def test_logging_helper_skips_none_layer_number():
    from torchtitan_npu.models.common.dsa_indexer_loss import (
        DSAIndexerLossLoggingHelper,
    )

    DSAIndexerLossLoggingHelper.tracker.clear()
    loss = torch.tensor(0.123)
    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
        loss, layer_number=None, num_layers=4
    )
    assert "values" not in DSAIndexerLossLoggingHelper.tracker
