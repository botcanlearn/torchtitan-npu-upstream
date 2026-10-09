"""Shared execution adapter for OverrideDefinitions-based 8P/16P/64P cases.

TorchTitan's run_tests()/GPUPool remains the 1-node 8P test runner.
Cross-node orchestration stays in Lite Actions; only the per-node recipe and
TensorBoard assertions live here, never SSH/resource scheduling.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.loss_compare import extract_losses_from_tensorboard


def required_steps() -> int:
    try:
        steps = int(os.environ.get("STEPS", "5"))
    except ValueError as exc:
        raise ValueError("STEPS must be a positive integer") from exc
    if steps < 1:
        raise ValueError("STEPS must be positive")
    return steps


def validate_assets(parser: argparse.ArgumentParser, *, ckpt_required: bool = False) -> None:
    paths = ("HF_ASSETS_PATH", "CKPT_INIT_LOAD_PATH") if ckpt_required else ("HF_ASSETS_PATH",)
    for key in paths:
        value = os.environ.get(key)
        if not value or not Path(value).is_dir():
            parser.error(f"{key} must point to an existing directory")


def select_definition(cases: list[OverrideDefinitions]) -> OverrideDefinitions:
    """Select one registered definition; future files may contain many tests."""
    selected = os.environ.get("LITE_CI_TEST_ID")
    if not selected and len(cases) == 1:
        return cases[0]
    matching = [item for item in cases if item.test_name == selected]
    if len(matching) != 1:
        raise ValueError("requested test definition is not unique in this module")
    return matching[0]


def run_single(test: OverrideDefinitions) -> None:
    from tests.integration_tests.run_tests import run_tests

    parser = argparse.ArgumentParser(description=test.test_descr)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    validate_assets(parser)
    if os.environ.get("COMPILE_ENABLE", "1") not in ("0", "1"):
        parser.error("COMPILE_ENABLE must be 0 or 1")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output_dir must be empty")
    run_tests(argparse.Namespace(output_dir=str(output), ngpu=test.ngpu,
                                 test_name=test.test_name, exclude=None),
              [test], parallel=True)


def run_distributed(test: OverrideDefinitions, *, nnodes: int,
                    ckpt_required: bool = False) -> None:
    parser = argparse.ArgumentParser(description=test.test_descr)
    parser.add_argument("mode", choices=("launch", "verify"))
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    steps = required_steps()
    run_root = args.output_dir / test.test_name / "test_run"
    if args.mode == "verify":
        values = extract_losses_from_tensorboard(run_root, "tb_phase_0")
        expected = set(range(1, steps + 1))
        if set(values) != expected:
            raise RuntimeError(f"{test.test_name}: expected steps {sorted(expected)}, got {sorted(values)}")
        print(f"[MULTINODE_VERIFY] PASS nodes={nnodes} ngpu={test.ngpu} "
              f"steps={sorted(values)} loss={values}", flush=True)
        return

    validate_assets(parser, ckpt_required=ckpt_required)
    ips = [x.strip() for x in os.environ.get("NODE_IPS", "").split(",")]
    if len(ips) != nnodes or any(not x for x in ips):
        parser.error(f"NODE_IPS requires exactly {nnodes} nonempty IPs")
    if os.environ.get("NGPU", str(test.ngpu)) != str(test.ngpu):
        parser.error(f"NGPU must be {test.ngpu} per node")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output_dir must be empty")
    command = ["bash", test.train_script, "--dump_folder", str(run_root),
               *test.train_args, "--metrics.save_tb_folder=tb_phase_0"]
    print("[MULTINODE_LAUNCH] " + " ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, env={**os.environ, **test.env_vars}))
