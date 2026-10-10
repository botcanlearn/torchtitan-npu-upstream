"""A5 64P Pro case; eight nodes use the existing 32P Pro recipe."""
from __future__ import annotations
import os
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.nightly_all_models_test.runner import run_distributed

def build_test_list()->list[OverrideDefinitions]:
    steps=int(os.environ.get("LITE_TEST_STEPS","5"))
    if not 1<=steps<=1000:raise ValueError("invalid steps")
    return [OverrideDefinitions(
        test_name="dsv4_pro_a5_64p",test_descr="A5 64P Pro smoke",
        ngpu=8,nnodes=8,ckpt_init_required=True,
        train_script="examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh",
        train_args=("--metrics.enable_tensorboard","--metrics.log_freq=1"),
        override_args=[("--training.steps",str(steps))],
        env_vars={"MODULE":"torchtitan_npu.models.deepseek_v4",
                  "CONFIG":"deepseek_v4_pro_61layers_32experts"},
        expected_steps=(tuple(range(1,steps+1)),),
        use_golden=False,check_loss=False,timeout=14200,
    )]
build_a5_64p_test_list=build_test_list

def main():
    run_distributed(build_test_list()[0],nnodes=8,ckpt_required=True)

if __name__=="__main__":main()
