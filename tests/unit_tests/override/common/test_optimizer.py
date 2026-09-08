# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for swap allocation and lazy optimizer-state initialization."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchtitan.config as torchtitan_config


with (
    patch.dict(
        sys.modules,
        {
            "torch_npu": MagicMock(),
            "torchtitan_npu.compile": MagicMock(),
            "torchtitan_npu.patches": MagicMock(),
        },
    ),
    patch.multiple(
        torchtitan_config,
        derive=lambda cfg, target: target,
        override=lambda **kwargs: lambda function: function,
        create=True,
    ),
):
    from torchtitan_npu.override.common import optimizer as product_optimizer

make_swap = vars(product_optimizer)["_make_swap"]
swap_state_init_hook = vars(product_optimizer)["_swap_state_init_hook"]

pytestmark = pytest.mark.cpu


@pytest.fixture
def swap_allocations(monkeypatch):
    allocations = []

    def allocate(size, *, dtype, device):
        allocations.append((torch.Size(size), dtype, torch.device(device)))
        return torch.empty(size, dtype=dtype, device=device)

    monkeypatch.setattr(product_optimizer.torch_npu, "empty_with_swapped_memory", allocate)
    return allocations


def test_make_swap_allocates_nonempty_tensor(swap_allocations):
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    result = make_swap(source)

    assert swap_allocations == [(source.size(), source.dtype, source.device)]
    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert result.data_ptr() != source.data_ptr()


def test_make_swap_uses_regular_empty_tensor_for_zero_size_input(swap_allocations):
    source = torch.empty(0, 4)

    result = make_swap(source)

    assert result.shape == source.shape
    assert result.numel() == 0
    assert swap_allocations == []


def test_swap_state_hook_initializes_only_missing_states_with_gradients(
    swap_allocations,
):
    missing = torch.nn.Parameter(torch.ones(2))
    existing = torch.nn.Parameter(torch.ones(2))
    no_grad = torch.nn.Parameter(torch.ones(2))
    missing.grad = torch.ones_like(missing)
    existing.grad = torch.ones_like(existing)
    sentinel = object()
    state = {missing: {}, existing: {"sentinel": sentinel}, no_grad: {}}
    optimizer = types.SimpleNamespace(
        param_groups=[{"params": [missing, existing, no_grad]}],
        state=state,
    )

    swap_state_init_hook(optimizer, (), {})

    assert set(state[missing]) == {"step", "exp_avg", "exp_avg_sq"}
    assert state[missing]["step"].item() == 0
    assert state[missing]["step"].dtype == torch.float32
    assert torch.count_nonzero(state[missing]["exp_avg"]) == 0
    assert torch.count_nonzero(state[missing]["exp_avg_sq"]) == 0
    assert state[existing] == {"sentinel": sentinel}
    assert state[no_grad] == {}
    assert swap_allocations == [(missing.size(), missing.dtype, missing.device)] * 2


def test_virtual_derives_optimizer_config():
    cfg = object()

    assert product_optimizer.virtual(cfg) is product_optimizer.VirtualOptimizersContainer.Config
