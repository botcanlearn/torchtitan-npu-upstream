"""A3 16P nightly model Integration Test cases (2 nodes x 8 NPU)."""
from __future__ import annotations
import os
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.nightly_all_models_test.runner import required_steps, run_distributed, select_definition


def build_a3_16p_test_list() -> list[OverrideDefinitions]:
    steps = required_steps()
    return [OverrideDefinitions(
        test_name="dsv4_flash_a3_16p_example",
        test_descr="DeepSeek-V4 Flash A3 16P nightly distributed E2E",
        ngpu=8,  # local NPU count; world_size is 2*8
        train_script="examples/deepseek_v4/debug/deepseek_v4_flash_16p_cpt_4k_a3.sh",
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        override_args=[()],
        env_vars={
            "MODULE": "torchtitan_npu.models.deepseek_v4",
            "CONFIG": "deepseek_v4_flash_43layers_16experts",
            "HF_ASSETS_PATH": os.environ.get("HF_ASSETS_PATH", ""),
            "STEPS": str(steps),
            "COMPILE_ENABLE": os.environ.get("COMPILE_ENABLE", "0"),
            "USE_GOLDEN": "0",
        },
        expected_steps=(tuple(range(1, steps + 1)),),
        use_golden=False, check_loss=False, timeout=7200,
    )]


def main() -> None:
    run_distributed(select_definition(build_a3_16p_test_list()), nnodes=2)


if __name__ == "__main__":
    main()
