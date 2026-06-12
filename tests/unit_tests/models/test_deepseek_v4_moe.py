# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit test for DeepSeek-V4 MoE buffer initialization.

The bug under test is pure PyTorch logic: moe.py depends only on torch and
torchtitan, so it is loaded directly by file path instead of through the
torchtitan_npu package, which requires torch_npu and applies patches on
import.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchtitan", reason="upstream torchtitan is required")

_MODULE_NAME = "torchtitan_npu.models.deepseek_v4.moe"


def _load_moe_module():
    if _MODULE_NAME in sys.modules:  # already imported, e.g. by the NPU CI
        return sys.modules[_MODULE_NAME]
    repo_root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        repo_root / "torchtitan_npu" / "models" / "deepseek_v4" / "moe.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_moe_module = _load_moe_module()
NUM_EXPERTS = 8


def test_init_weights_zeroes_moe_buffers_after_to_empty():
    """init_weights must reset MoE buffers left uninitialized by to_empty()."""
    config = _moe_module.MoE.Config(
        moe_args=_moe_module.MoEArgs(
            num_experts=NUM_EXPERTS,
            num_shared_experts=1,
            top_k=2,
            load_balance_coeff=1e-3,
            n_hash_layers=0,
        ),
        dim=16,
        hidden_dim=32,
        layer_id=0,
        vocab_size=128,
    )
    with torch.device("meta"):
        moe = config.build()
    moe.to_empty(device="cpu")
    # simulate the garbage memory to_empty() leaves behind
    with torch.no_grad():
        moe.tokens_per_expert.fill_(float("nan"))
        moe.expert_bias.fill_(1e30)

    moe.init_weights(0.02, torch.device("cpu"))

    assert torch.equal(moe.tokens_per_expert, torch.zeros(NUM_EXPERTS))
    assert torch.equal(moe.expert_bias, torch.zeros(NUM_EXPERTS))
