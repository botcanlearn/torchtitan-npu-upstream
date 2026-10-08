#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the A3 8-NPU DeepSeek-V4 Flash example through the integration runner."""

import argparse
import os
from pathlib import Path

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.run_tests import run_tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Empty directory for test artifacts")
    args = parser.parse_args()

    hf_assets_path = os.environ.get("HF_ASSETS_PATH")
    if not hf_assets_path or not Path(hf_assets_path).is_dir():
        parser.error("HF_ASSETS_PATH must point to an existing tokenizer directory")

    try:
        steps = int(os.environ.get("STEPS", "5"))
    except ValueError:
        parser.error("STEPS must be a positive integer")
    if steps < 1:
        parser.error("STEPS must be a positive integer")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output_dir must be empty")

    test = OverrideDefinitions(
        test_name="dsv4_flash_a3_8p_example",
        test_descr="DeepSeek-V4 Flash A3 8P example E2E",
        ngpu=8,
        train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh",
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        override_args=[()],
        env_vars={
            "MODULE": "torchtitan_npu.models.deepseek_v4",
            "CONFIG": "deepseek_v4_flash_43layers_16experts",
            "HF_ASSETS_PATH": hf_assets_path,
            "STEPS": str(steps),
            "USE_GOLDEN": "0",
        },
        expected_steps=(tuple(range(1, steps + 1)),),
        use_golden=False,
        check_loss=False,
        timeout=7200,
    )
    run_tests(
        argparse.Namespace(
            output_dir=str(args.output_dir),
            ngpu=8,
            test_name=test.test_name,
            exclude=None,
        ),
        [test],
        parallel=True,
    )


if __name__ == "__main__":
    main()
