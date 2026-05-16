# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torchtitan_npu.patches.optimizer import swap_optimizer


def test_unwrap_dtensor_returns_plain_tensor_for_non_dtensor():
    tensor = torch.randn(2, 2)

    result = swap_optimizer.unwrap_dtensor(tensor)

    assert result is tensor


def test_swap_optimizer_patch_step_function_updates_adam_classes(monkeypatch):
    orig_adam_step = torch.optim.Adam.step
    orig_adamw_step = torch.optim.AdamW.step
    try:
        monkeypatch.setattr(torch.optim.Adam, "step", lambda self, closure=None: None)
        monkeypatch.setattr(
            torch.optim.AdamW,
            "step",
            lambda self, closure=None: None,
        )

        swap_optimizer.patch_optimizer_step()

        assert torch.optim.Adam.step is swap_optimizer.swap_optimizer_step
        assert torch.optim.AdamW.step is swap_optimizer.swap_optimizer_step
    finally:
        monkeypatch.setattr(torch.optim.Adam, "step", orig_adam_step)
        monkeypatch.setattr(torch.optim.AdamW, "step", orig_adamw_step)


def test_swap_optimizer_config_build_rejects_unknown_optimizer():
    optimizer_config = swap_optimizer.SwapOptimizersContainer.Config(
        swap_optimizer=True,
        name="SGD",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="fused",
        swap_optimizer_times=8,
    )

    try:
        optimizer_config.build(model_parts=[])
        raise AssertionError("Expected NotImplementedError for unsupported optimizer")
    except NotImplementedError as exc:
        assert "Optimizer SGD not added" in str(exc)


def test_swap_optimizer_config_build_uses_swap_container(monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, closure=None: None)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, closure=None: None)

    model = torch.nn.Linear(4, 4)
    optimizer_config = swap_optimizer.SwapOptimizersContainer.Config(
        swap_optimizer=True,
        name="AdamW",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="fused",
        swap_optimizer_times=16,
    )

    container = optimizer_config.build(model_parts=[model])

    assert isinstance(container, swap_optimizer.SwapOptimizersContainer)
    assert len(container.optimizers) == 1
    assert container.optimizers[0].step.__func__ is swap_optimizer.swap_optimizer_step
