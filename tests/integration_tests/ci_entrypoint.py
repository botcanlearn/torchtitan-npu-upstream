#!/usr/bin/env python3
"""Trusted CI entrypoint. Only tests registered in this model commit may run.

The dispatcher passes a plain test ID, NEVER an arbitrary Python module, shell
snippet or command supplied by GitHub workflow inputs.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import runpy
import sys

REGISTRY = Path(__file__).with_name("ci_registry.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("launch", "verify"))
    parser.add_argument("test_id")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if args.test_id not in registry:
        parser.error("test_id not registered in the current model commit")
    profile = registry[args.test_id]
    module = profile["module"]
    if not (isinstance(module, str) and re.fullmatch(
            r"tests\.integration_tests\.nightly_all_models_test\.[a-z][a-z0-9_]*_tests",
            module)):
        parser.error("invalid registered test module")
    mode = profile["mode"]
    if args.phase == "verify" and mode == "single":
        # run_tests already validates TensorBoard and steps in its launch phase.
        return
    os.environ["LITE_CI_TEST_ID"] = args.test_id
    sys.argv = [module, *([args.phase] if mode == "distributed" else []), str(args.output_dir)]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
