"""DeepSeek V4 Flash A3 16P Eager test (2 x 8 NPUs)."""
from __future__ import annotations
import os
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.lite_actions.nightly_all_models_test.runner import run_distributed

# The upstream recipe enables Muon swap by default; Tyro uses the final
# --override.imports list, so replace it with the baseline NPU operations
# (without optimizer.swap_optimizer). No shell environment branch is required.
NPU_IMPORTS=(
 "torchtitan_npu.override.common.rms_norm.asc",
 "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata",
 "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li",
 "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata",
 "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
 "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
 "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
 "torchtitan_npu.override.common.token_dispatcher.asc",
 "torchtitan_npu.override.common.rope.asc_complex",
)

def build_test_list() -> list[OverrideDefinitions]:
    steps=int(os.environ.get("LITE_TEST_STEPS","5"))
    if not 1<=steps<=1000:raise ValueError("invalid steps")
    return [OverrideDefinitions(
        test_name="dsv4_flash_a3_16p_example",test_descr="A3 two-node Flash Eager",
        ngpu=8,nnodes=2,
        train_script="examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh",
        train_args=("--metrics.enable_tensorboard","--metrics.log_freq=1"),
        override_args=[(
          "--parallelism.expert-parallel-degree","16",
          "--parallelism.data-parallel-shard-degree","16",
          "--parallelism.data-parallel-replicate-degree","1",
          "--training.global-batch-size","128",
          "--training.steps",str(steps),
          "--optimizer.name","AdamW","--compile.no-enable",
          "--checkpoint.no-enable","--debug.moe-force-load-balance",
          "--comm.init-timeout-seconds","600",
          "--override.imports",*NPU_IMPORTS,
        )],
        env_vars={"MODULE":"torchtitan_npu.models.deepseek_v4",
                  "CONFIG":"deepseek_v4_flash_43layers_16experts"},
        expected_steps=(tuple(range(1,steps+1)),),
        use_golden=False,check_loss=False,timeout=7200,
    )]

build_a3_16p_test_list=build_test_list

def main():
    run_distributed(build_test_list()[0],nnodes=2)

if __name__=="__main__":main()
