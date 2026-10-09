#!/usr/bin/env python3
"""Trusted CI entrypoint. Only tests registered in this model commit may run.

The dispatcher passes a plain test ID, NEVER an arbitrary Python module, shell
snippet or command supplied by GitHub workflow inputs.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import importlib

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
            r"tests\.integration_tests(?:\.[a-z][a-z0-9_]*){1,6}", module)):
        parser.error("invalid registered integration test module")
    mode = profile["mode"]
    if args.phase == "verify" and mode == "single":
        return  # The single-node suite verifies steps during launch.
    if mode not in ("single", "distributed") or type(profile.get("nnodes")) is not int:
        parser.error("invalid registered test mode or topology")
    definitions = importlib.import_module(module).build_test_list()
    matching = [case for case in definitions if case.test_name == args.test_id]
    if len(matching) != 1:
        parser.error("test_id is not uniquely defined by registered builder")
    from tests.integration_tests.nightly_all_models_test.runner import run_single, run_distributed
    test = matching[0]
    if test.ngpu != profile["ngpu"]:
        parser.error("test definition and registered GPU count disagree")
    if mode == "single":
        if profile["nnodes"] != 1:
            parser.error("single-node test must have nnodes=1")
        run_single(test, output_dir=args.output_dir)
    else:
        run_distributed(test, nnodes=profile["nnodes"],
                        ckpt_required=bool(profile.get("ckpt_init_required", False)),
                        phase=args.phase, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
