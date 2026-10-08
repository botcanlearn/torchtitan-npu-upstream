#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""A3 DeepSeek-V4 Flash 16P: per-node launch and master TensorBoard verification.

The existing run_tests NPU pool is intentionally NOT used: it allocates 16 NPUs
on one machine, whereas this test requires 8 physical NPUs on each of two hosts.
The dispatcher starts 'launch' concurrently on both nodes, then 'verify' on
the master only after both node processes exit 0.
"""

import argparse
import os
import subprocess
from pathlib import Path

from tests.integration_tests.loss_compare import extract_losses_from_tensorboard

EXAMPLE = "examples/deepseek_v4/debug/deepseek_v4_flash_16p_cpt_4k_a3.sh"
NAME = "dsv4_flash_a3_16p_example"


def build_launch_command(output_dir: Path) -> list[str]:
    return [
        "bash", EXAMPLE, "--dump_folder", str(output_dir / NAME / "test_run"),
        "--metrics.enable_tensorboard", "--metrics.log_freq=1",
        "--metrics.save_tb_folder=tb_phase_0",
    ]


def verify(output_dir: Path, steps: int) -> None:
    values = extract_losses_from_tensorboard(
        output_dir / NAME / "test_run", "tb_phase_0"
    )
    expected = set(range(1, steps + 1))
    if set(values) != expected:
        raise RuntimeError(f"{NAME}: expected steps {sorted(expected)}, got {sorted(values)}")
    print(f"[16P_VERIFY] PASS world_size=16, steps={sorted(values)}, loss={values}", flush=True)


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
        parser.error("STEPS must be a positive integer")
    if args.mode == "verify":
        verify(args.output_dir, steps)
        return
    assets = os.environ.get("HF_ASSETS_PATH")
    if not assets or not Path(assets).is_dir():
        parser.error("HF_ASSETS_PATH must point to a tokenizer directory")
    nodes = os.environ.get("NODE_IPS", "").split(",")
    if len(nodes) != 2 or any(not x.strip() for x in nodes):
        parser.error("NODE_IPS must list exactly two node IPs")
    if os.environ.get("NGPU", "8") != "8":
        parser.error("NGPU must be 8 per host")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output_dir must be empty")
    cmd = build_launch_command(args.output_dir)
    print("[16P_LAUNCH] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=os.environ.copy()))


if __name__ == "__main__":
    main()
