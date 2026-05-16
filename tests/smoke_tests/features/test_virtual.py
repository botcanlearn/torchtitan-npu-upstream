# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torchtitan_npu.patches.optimizer.virtual_optimizer import virtual_optimizer_step

pytestmark = pytest.mark.smoke


def test_virtual_optimizer_step_can_be_patched_on_adam_classes(monkeypatch):
    orig_adam_step = torch.optim.Adam.step
    orig_adamw_step = torch.optim.AdamW.step
    try:
        monkeypatch.setattr(torch.optim.Adam, "step", lambda self, closure=None: None)
        monkeypatch.setattr(
            torch.optim.AdamW,
            "step",
            lambda self, closure=None: None,
        )

        torch.optim.Adam.step = virtual_optimizer_step
        torch.optim.AdamW.step = virtual_optimizer_step

        assert torch.optim.Adam.step is virtual_optimizer_step
        assert torch.optim.AdamW.step is virtual_optimizer_step
    finally:
        monkeypatch.setattr(torch.optim.Adam, "step", orig_adam_step)
        monkeypatch.setattr(torch.optim.AdamW, "step", orig_adamw_step)
