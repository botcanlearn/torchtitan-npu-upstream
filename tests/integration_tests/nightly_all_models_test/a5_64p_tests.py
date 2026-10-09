"""A5 64P nightly model Integration Test cases (8 nodes x 8 NPU)."""
from __future__ import annotations
import os
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.nightly_all_models_test.runner import required_steps, run_distributed


def build_a5_64p_test_list() -> list[OverrideDefinitions]:
    steps = required_steps()
    return [OverrideDefinitions(
        test_name="dsv4_pro_a5_64p",
        test_descr="DeepSeek-V4 Pro A5 64P nightly distributed E2E",
        ngpu=8,  # local; total 8x8
        train_script="examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh",
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        override_args=[()],
        env_vars={
            "MODULE": "torchtitan_npu.models.deepseek_v4",
            "CONFIG": "deepseek_v4_pro_61layers_32experts",
            "HF_ASSETS_PATH": os.environ.get("HF_ASSETS_PATH", ""),
            "STEPS": str(steps), "NGPU": "8", "EP": "32",
            "DP_SHARD": "32", "GBS": "256", "USE_GOLDEN": "0",
        },
        expected_steps=(tuple(range(1, steps + 1)),),
        use_golden=False, check_loss=False, timeout=14400,
    )]


# Shared CI entrypoint imports this stable contract; the original builder remains public.
build_test_list = build_a5_64p_test_list

def main() -> None:
    run_distributed(build_a5_64p_test_list()[0], nnodes=8, ckpt_required=True)


if __name__ == "__main__":
    main()
