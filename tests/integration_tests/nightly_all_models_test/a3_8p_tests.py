"""A3 8P nightly model Integration Test cases (single execution node)."""
from __future__ import annotations
import os
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.nightly_all_models_test.runner import required_steps, run_single


def build_a3_8p_test_list() -> list[OverrideDefinitions]:
    steps = required_steps()
    return [OverrideDefinitions(
        test_name="dsv4_flash_a3_8p_example",
        test_descr="DeepSeek-V4 Flash A3 8P nightly E2E",
        ngpu=8,
        train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh",
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        override_args=[()],
        env_vars={
            "MODULE": "torchtitan_npu.models.deepseek_v4",
            "CONFIG": "deepseek_v4_flash_43layers_16experts",
            "HF_ASSETS_PATH": os.environ.get("HF_ASSETS_PATH", ""),
            "STEPS": str(steps),
            "COMPILE_ENABLE": os.environ.get("COMPILE_ENABLE", "1"),
            "USE_GOLDEN": "0",
        },
        expected_steps=(tuple(range(1, steps + 1)),),
        use_golden=False, check_loss=False, timeout=7200,
    )]


def main() -> None:
    run_single(build_a3_8p_test_list()[0])


if __name__ == "__main__":
    main()
