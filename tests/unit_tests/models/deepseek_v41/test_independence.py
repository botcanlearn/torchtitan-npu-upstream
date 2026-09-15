# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 dependency and import-side-effect boundaries."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
V4_MODULE = re.compile(r"torchtitan_npu\.(?:models|override)\.deepseek_v4(?:\b|\.)")


def test_v41_package_has_no_v4_references():
    paths = list((REPO / "torchtitan_npu/models/deepseek_v41").rglob("*.py"))
    paths += list((REPO / "tests/unit_tests/models/deepseek_v41").rglob("*.py"))
    paths += [
        REPO / "tests/integration_tests/deepseek_v41.py",
        REPO / "examples/deepseek_v41/debug/deepseek_v41_flash_8p_cpt_4k_a3.sh",
    ]
    for path in paths:
        if path == Path(__file__):
            continue
        source = path.read_text(encoding="utf-8")
        assert not V4_MODULE.search(source), path
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        assert not V4_MODULE.search(f"{node.module}.{alias.name}"), path


def _run_isolated(code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "1", "PYTHONPATH": str(REPO)},
        timeout=180,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"


def test_v41_builds_and_runs_without_v4():
    _run_isolated(r"""
import sys

class Blocker:
    blocked = ("torchtitan_npu.models.deepseek_v4", "torchtitan_npu.override.deepseek_v4")

    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in self.blocked):
            raise ImportError(f"independence blocker: {name}")
        return None

sys.meta_path.insert(0, Blocker())
from dataclasses import replace
import importlib
import torch
registry = importlib.import_module("torchtitan_npu.models.deepseek_v41.model_registry")
registry._DEBUG_WIDTHS = replace(
    registry._DEBUG_WIDTHS, dim=8, n_heads=2, head_dim=8, rope_head_dim=4,
    q_lora_rank=8, o_lora_rank=4, n_groups=1, index_n_heads=2,
    index_head_dim=4, moe_inter_dim=16, vision_dim=8, vision_heads=2, vision_inter_dim=16,
)
cfg = registry.model_registry("deepseek_v41_debugmodel").model
cfg.vocab_size = cfg.tok_embeddings.num_embeddings = cfg.lm_head.out_features = 64
from tests.unit_tests.models.mtp_test_utils import build_cpu_model
model = build_cpu_model(cfg)
tokens = torch.arange(128).remainder(32).unsqueeze(0)
_, _, kwargs = model.build_attention_masks(tokens, tokens, {"positions": torch.arange(128).unsqueeze(0)})
out = model(tokens, **kwargs)
assert torch.isfinite(out).all()
out.square().mean().backward()
assert len(model.layers) == 40
for layer in model.layers.values():
    assert layer.moe.tokens_per_expert_E.sum() == 128 * 6
    assert layer.moe.routed_experts.inner_experts.w1_EFD.grad is not None
""")


def test_v41_import_does_not_patch_v4():
    _run_isolated(r"""
import importlib
import sys
import torch.nn.functional as functional
v4 = importlib.import_module("torchtitan_npu.models.deepseek_v4")
moe = importlib.import_module("torchtitan.models.common.moe")
assert "torchtitan_npu.models.deepseek_v41" not in sys.modules

def identities():
    return (v4.attention.Attention, v4.attention.Attention.forward,
            v4.model.DeepSeekV4Model, v4.model.DeepSeekV4Model.forward,
            moe.MoE, moe.MoE.forward, moe.TokenChoiceTopKRouter,
            moe.TokenChoiceTopKRouter.forward, functional.cross_entropy)

def config_contract():
    cfg = v4.model_registry("debugmodel").model
    return [(type(layer.attention), type(layer.moe), type(layer.moe.router),
             layer.attention.compress_ratio, layer.moe.router.score_func)
            for layer in cfg.layers]

before, config_before = identities(), config_contract()
v41 = importlib.import_module("torchtitan_npu.models.deepseek_v41")
v41.model_registry("deepseek_v41_debugmodel")
assert identities() == before
assert config_contract() == config_before
""")
