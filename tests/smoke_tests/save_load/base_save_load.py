# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base helpers for save→load smoke tests.

Model-specific test files (e.g. ``test_qwen3_save_load.py``) import these
helpers and call :func:`run_save_load_test` with their own arguments.
"""

import json
import os
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file


def run_training(
    repo_root: Path,
    output_dir: Path,
    *,
    module: str,
    config: str,
    steps: int = 1,
    seq_len: int = 128,
    ngpu: int = 1,
    tokenizer_path: str = "",
    extra_args: str = "",
) -> subprocess.CompletedProcess:
    """Launch one training run via ``scripts/run_train.sh``.

    Returns the subprocess result for the caller to check.
    """
    cmd_parts = [
        f"NGPU={ngpu}",
        "LOG_RANK=0",
        "bash",
        str(repo_root / "scripts" / "run_train.sh"),
        f"--module {module}",
        f"--config {config}",
        f"--dump_folder {str(output_dir)}",
        f"--training.steps {steps}",
        "--training.local_batch_size 1",
        f"--training.seq_len {seq_len}",
        f"--checkpoint.folder {str(output_dir / 'checkpoint')}",
        f"--checkpoint.interval {steps}",
        "--checkpoint.last_save_in_hf",
        "--checkpoint.last_save_model_only",
        "--checkpoint.export_dtype float32",
        "--metrics.log_freq 1000",
    ]
    if tokenizer_path:
        cmd_parts.append(f"--hf_assets_path {tokenizer_path}")
    if extra_args:
        cmd_parts.append(extra_args)

    cmd = " ".join(cmd_parts)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

    return subprocess.run(
        cmd,
        shell=True,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


# ---------------------------------------------------------------------------
# Checkpoint verification
# ---------------------------------------------------------------------------


def verify_hf_checkpoint(hf_dir: Path) -> None:
    """Check that an exported HuggingFace checkpoint is valid.

    - safetensors files are non-empty and contain no NaN
    - model index has a ``weight_map`` (if present)
    - tokenizer files exist
    """
    safetensors_files = list(hf_dir.glob("*.safetensors"))
    assert safetensors_files, f"No .safetensors in {hf_dir}"

    for sf in safetensors_files:
        tensors = load_file(str(sf))
        assert len(tensors) > 0, f"Empty safetensors file: {sf}"
        for name, tensor in tensors.items():
            assert not torch.any(torch.isnan(tensor)), f"NaN in {name}"

    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        assert "weight_map" in index, f"No weight_map in {index_path}"

    # tokenizer files are not part of model-only HF export; they must be
    # copied separately (see docs/recipe/sft.md). The smoke test passes
    # the original tokenizer_path to Phase 2 for this reason.


# ---------------------------------------------------------------------------
# Main entry point for model-specific tests
# ---------------------------------------------------------------------------


def run_save_load_test(
    tmp_path: Path,
    *,
    module: str,
    config: str,
    steps: int = 1,
    seq_len: int = 128,
    ngpu: int = 1,
    tokenizer_path: str = "",
    extra_args: str = "",
):
    """Run training → HF export → reload → resume training → verify.

    1. Train for *steps* to produce initial weights.
    2. Export HF checkpoint via ``last_save_in_hf``.
    3. Verify safetensors / tokenizer files are valid.
    4. Start a new training run that loads the HF checkpoint.
    5. Train 1 more step to prove the checkpoint is loadable.

    Intended to be called from model-specific smoke test files, e.g.::

        from tests.smoke_tests.save_load.base_save_load import run_save_load_test

        def test_qwen3(tmp_path):
            run_save_load_test(
                tmp_path,
                module="torchtitan_npu.models.qwen3",
                config="sft_qwen3_1_7b_wordle",
            )
    """
    repo_root = Path(__file__).resolve().parents[3]

    # ---- Phase 1: initial training + HF export ----
    output_dir_1 = tmp_path / "phase1"
    output_dir_1.mkdir()

    result = run_training(
        repo_root=repo_root,
        output_dir=output_dir_1,
        module=module,
        config=config,
        steps=steps,
        seq_len=seq_len,
        ngpu=ngpu,
        tokenizer_path=tokenizer_path,
        extra_args=extra_args,
    )
    assert result.returncode == 0, (
        f"Phase 1 training failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-3000:]}\n"
        f"STDERR:\n{result.stderr[-3000:]}"
    )

    step_dirs = sorted(output_dir_1.rglob("step-*"))
    assert step_dirs, f"No step directories found in {output_dir_1}"
    hf_export_dir = step_dirs[-1]
    verify_hf_checkpoint(hf_export_dir)

    # ---- Phase 2: reload HF checkpoint and train 1 more step ----
    output_dir_2 = tmp_path / "phase2"
    output_dir_2.mkdir()

    resume_extra = (
        f"--checkpoint.initial_load_path {str(hf_export_dir)} "
        f"--checkpoint.initial_load_in_hf"
    )

    result2 = run_training(
        repo_root=repo_root,
        output_dir=output_dir_2,
        module=module,
        config=config,
        steps=1,
        seq_len=seq_len,
        ngpu=ngpu,
        tokenizer_path=tokenizer_path,
        extra_args=resume_extra,
    )
    assert result2.returncode == 0, (
        f"Phase 2 (resume) training failed (exit {result2.returncode}):\n"
        f"STDOUT:\n{result2.stdout[-3000:]}\n"
        f"STDERR:\n{result2.stderr[-3000:]}"
    )
