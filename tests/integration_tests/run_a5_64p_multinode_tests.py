#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the BSD-style license in the repository root.
"""A5 Pro 64P (8 nodes x 8 devices), launch on each node, verify on master.

The dispatcher owns SSH, devices, staging, retry/cleanup and reporting;
this module owns only the DeepSeek-V4 Pro model test and TensorBoard checks.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from tests.integration_tests.loss_compare import extract_losses_from_tensorboard

EXAMPLE = "examples/deepseek_v4/debug/deepseek_v4_pro_64p_cpt_4k_a5.sh"
NAME = "dsv4_pro_a5_64p"


def launch_command(output_dir: Path) -> list[str]:
    return ["bash", EXAMPLE, "--dump_folder", str(output_dir / NAME / "test_run"),
            "--metrics.enable_tensorboard", "--metrics.log_freq=1",
            "--metrics.save_tb_folder=tb_phase_0"]


def verify(output_dir: Path, steps: int) -> None:
    values = extract_losses_from_tensorboard(output_dir / NAME / "test_run", "tb_phase_0")
    expected = set(range(1, steps + 1))
    if set(values) != expected:
        raise RuntimeError(f"{NAME}: expected {sorted(expected)}, got {sorted(values)}")
    print(f"[64P_VERIFY] PASS steps={sorted(values)} loss={values}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "verify"))
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        steps = int(os.environ.get("STEPS", "5"))
    except ValueError:
        parser.error("STEPS must be a positive integer")
    if steps < 1:
        parser.error("STEPS must be positive")
    if args.mode == "verify":
        verify(args.output_dir, steps)
        return
    if len([ip for ip in os.environ.get("NODE_IPS", "").split(",") if ip.strip()]) != 8:
        parser.error("A5 64P requires 8 NODE_IPS")
    if os.environ.get("NGPU") != "8":
        parser.error("A5 64P requires NGPU=8 per host")
    for name in ("HF_ASSETS_PATH", "CKPT_INIT_LOAD_PATH"):
        folder = os.environ.get(name)
        if not folder or not Path(folder).is_dir():
            parser.error(f"{name} must be an existing directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output_dir must be empty")
    raise SystemExit(subprocess.call(launch_command(args.output_dir), env=os.environ.copy()))


if __name__ == "__main__":
    main()
