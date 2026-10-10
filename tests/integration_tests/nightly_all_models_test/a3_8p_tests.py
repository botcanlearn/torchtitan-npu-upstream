"""A3 8P DeepSeek-V4 Flash Eager regression: Muon and AdamW."""
from tests.integration_tests import OverrideDefinitions

STEPS = 5
EXPECTED_STEPS = tuple(range(1, STEPS + 1))
SCRIPT = "examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh"

# These paths are testcase-owned: upgrading the toolkit/model never changes
# Lite Actions deployment config or dispatcher logic.
ASSET_ENV = {
    "ASCEND_SET_ENV_PATH": "/mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh",
    "HF_ASSETS_PATH": "/mnt/share/models/DeepSeek-V4-Flash-bf16",
    "CKPT_INIT_LOAD_PATH": "/mnt/share/dsv4_ckpt_8rank",
}


def build_test_list() -> list[OverrideDefinitions]:
    common = dict(
        ngpu=8,
        nnodes=1,
        train_script=SCRIPT,
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        expected_steps=(EXPECTED_STEPS,),
        check_loss=False,
        timeout=7200,
    )
    return [
        OverrideDefinitions(
            test_name="dsv4_flash_a3_8p_example",
            test_descr="DeepSeek-V4 Flash A3 8P Muon Eager, 5 steps",
            override_args=[("--training.steps", str(STEPS), "--compile.no-enable")],
            env_vars=ASSET_ENV,
            **common,
        ),
        OverrideDefinitions(
            test_name="dsv4_flash_a3_8p_adamw",
            test_descr="DeepSeek-V4 Flash A3 8P AdamW Eager, 5 steps",
            override_args=[("--training.steps", str(STEPS), "--compile.no-enable",
                            "--optimizer.name", "AdamW")],
            env_vars={**ASSET_ENV, "OPTIMIZER_OVERRIDES": ""},
            **common,
        ),
    ]
