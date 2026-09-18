# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests import OverrideDefinitions

# DeepSeek-V4.1 keeps no default integration case: like the upstream
# torchtitan model it is covered by the CPU unit suite plus this explicit,
# manually selected fused smoke.  The 8-card reference and fused shapes stay
# available through run_train.sh with the CONFIG= recipes (see
# examples/deepseek_v4_1/readme.md); no dedicated loss anchor is kept and no
# A5 case is registered in an A3 smoke environment.


def build_deepseek_v4_1_fused_test_list() -> list[OverrideDefinitions]:
    """Explicit A3 multimodal packing/fusion/Muon smoke; outside the default CI pool."""
    ngpu = 2
    args = (
        "--dataloader.dataset-path=tests/assets/cc12m_test",
        "--hf-assets-path=tests/assets/deepseek_v3",
        "--training.steps=2",
        "--dataloader.packing-buffer-size=4",
        "--debug.seed=42",
        "--debug.deterministic",
        f"--training.global-batch-size={ngpu}",
        f"--parallelism.data-parallel-shard-degree={ngpu}",
        "--parallelism.data-parallel-replicate-degree=1",
        f"--parallelism.expert-parallel-degree={ngpu}",
        "--parallelism.tensor-parallel-degree=1",
        "--parallelism.context-parallel-degree=1",
        "--parallelism.pipeline-parallel-degree=1",
        "--parallelism.context-parallel-load-balancer=None",
        "--checkpoint.no-enable",
        "--comm.init-timeout-seconds=7200",
        "--comm.train-timeout-seconds=600",
    )
    args += ("--optimizer.name=Muon",)
    return [
        OverrideDefinitions(
            override_args=[args],
            test_descr="V4.1 multimodal A3 fused/Muon",
            test_name="dsv41_multimodal_muon_2p",
            ngpu=ngpu,
            env_vars={
                "MODULE": "torchtitan_npu.models.deepseek_v4_1",
                "CONFIG": "deepseek_v4_1_debugmodel_multimodal_a3",
            },
            use_golden=False,
            check_loss=False,
            expected_steps=((1, 2),),
            timeout=3600,
        )
    ]
