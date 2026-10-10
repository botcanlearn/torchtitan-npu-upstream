"""A3 8P real integration tests; build_test_list is the single source of truth."""
from __future__ import annotations
import os
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.lite_actions.nightly_all_models_test.runner import run_single


def build_test_list() -> list[OverrideDefinitions]:
    steps = int(os.environ.get("LITE_TEST_STEPS", "5"))
    if not 1 <= steps <= 1000: raise ValueError("invalid case step budget")
    def case(name: str, count: int) -> OverrideDefinitions:
        return OverrideDefinitions(
            test_name=name, test_descr=f"DeepSeek V4 Flash A3 8P Eager {count}-step E2E",
            ngpu=8, nnodes=1,
            train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh",
            train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
            override_args=[("--training.steps",str(count),"--compile.no-enable")],
            env_vars={"MODULE":"torchtitan_npu.models.deepseek_v4",
                      "CONFIG":"deepseek_v4_flash_43layers_16experts"},
            expected_steps=(tuple(range(1,count+1)),),
            use_golden=False, check_loss=False,timeout=7200,
        )
    return [case("dsv4_flash_a3_8p_example",steps),
            case("dsv4_flash_a3_8p_multicase",3)]

build_a3_8p_test_list=build_test_list

def main() -> None:
    from tests.integration_tests.lite_actions.nightly_all_models_test.runner import run_single
    import argparse
    p=argparse.ArgumentParser();p.add_argument("test_name");p.add_argument("output_dir")
    args=p.parse_args()
    matches=[c for c in build_test_list() if c.test_name==args.test_name]
    if len(matches)!=1:p.error("unknown test")
    run_single(matches[0],output_dir=__import__('pathlib').Path(args.output_dir))

if __name__=="__main__":main()
