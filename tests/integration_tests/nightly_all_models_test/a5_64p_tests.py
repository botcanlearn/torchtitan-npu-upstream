"""A5 64P nightly model Integration Test cases (8 nodes x 8 NPU)."""
from __future__ import annotations
import os
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.nightly_all_models_test.runner import required_steps, run_distributed


def validate_parallelism(*, world_size: int = 64, tp: int = 1, pp: int = 1,
                         cp: int = 1, dp_shard: int = 32, ep: int = 32) -> int:
    """Check the candidate EP32 / DP-shard32 mapping before hardware use."""
    if min(world_size, tp, pp, cp, dp_shard, ep) < 1:
        raise ValueError("all parallelism factors must be positive")
    divisor = tp * pp * cp * dp_shard
    if world_size % divisor or world_size % ep:
        raise ValueError("parallelism cannot fit the available world size")
    return world_size // divisor


def build_a5_64p_test_list() -> list[OverrideDefinitions]:
    steps = required_steps()
    if validate_parallelism() != 2:
        raise ValueError("A5 64P requires DP replicate 2")
    return [OverrideDefinitions(
        test_name="dsv4_pro_a5_64p",
        test_descr="DeepSeek-V4 Pro A5 64P nightly distributed E2E",
        ngpu=8,  # local; total 8x8
        train_script="examples/deepseek_v4/debug/deepseek_v4_pro_64p_cpt_4k_a5.sh",
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


def main() -> None:
    run_distributed(build_a5_64p_test_list()[0], nnodes=8, ckpt_required=True)


if __name__ == "__main__":
    main()
