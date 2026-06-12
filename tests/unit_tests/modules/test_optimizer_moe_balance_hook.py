# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torch import nn
from torchtitan.components import optimizer as optimizer_module
from torchtitan.components.optimizer import register_moe_load_balancing_hook


class _FakeMoE(nn.Module):
    def __init__(self, *, has_bias: bool, load_balance_coeff: float = 1e-3):
        super().__init__()
        self.load_balance_coeff = load_balance_coeff
        self.register_buffer("tokens_per_expert", torch.tensor([2.0, 4.0]))
        if has_bias:
            self.register_buffer("expert_bias", torch.zeros(2))


class _FakeBlock(nn.Module):
    def __init__(self, *, moe_enabled: bool, moe: nn.Module | None = None):
        super().__init__()
        self.moe_enabled = moe_enabled
        self.moe = moe


class _FakeModelPart(nn.Module):
    def __init__(self, blocks: dict[str, nn.Module]):
        super().__init__()
        self.layers = nn.ModuleDict(blocks)


class _FakeOptimizers:
    def __init__(self):
        self.pre_hook = None

    def register_step_pre_hook(self, hook):
        self.pre_hook = hook


def _build_parallel_dims_without_loss_mesh():
    return SimpleNamespace(get_optional_mesh=lambda _name: None)


def test_moe_balance_hook_skips_moe_without_expert_bias_attribute():
    assert (
        optimizer_module.register_moe_load_balancing_hook.__name__
        == "register_moe_load_balancing_hook_with_expert_bias_guard"
    )

    optimizers = _FakeOptimizers()
    model_part = _FakeModelPart(
        {
            "0": _FakeBlock(moe_enabled=True, moe=_FakeMoE(has_bias=True)),
            "1": _FakeBlock(moe_enabled=True, moe=_FakeMoE(has_bias=False)),
        }
    )

    register_moe_load_balancing_hook(
        optimizers=optimizers,
        model_parts=[model_part],
        parallel_dims=_build_parallel_dims_without_loss_mesh(),
    )

    assert optimizers.pre_hook is not None

    # before fix this raises AttributeError on second MoE (missing expert_bias)
    optimizers.pre_hook()

    assert torch.allclose(model_part.layers["0"].moe.tokens_per_expert, torch.zeros(2))
    assert torch.allclose(model_part.layers["1"].moe.tokens_per_expert, torch.zeros(2))
    assert torch.allclose(model_part.layers["0"].moe.expert_bias, torch.tensor([0.001, -0.001]))
