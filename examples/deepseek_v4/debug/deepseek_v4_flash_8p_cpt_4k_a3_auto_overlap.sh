#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=1
export TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER=0
export TORCH_NPU_USE_COMPATIBLE_IMPL=1

SEQ_LEN="${SEQ_LEN:-4096}"
GBS="${GBS:-8}"
STEPS="${STEPS:-20}"

bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer.sh \
    --training.seq-len "${SEQ_LEN}" \
    --training.global-batch-size "${GBS}" \
    --training.steps "${STEPS}" \
    --debug.moe-force-load-balance \
    --compile.ep_overlap.enabled \
    --compile.ep_overlap.chunk_dim seq \
    --compile.ep_overlap.strategy graph \
    --compile.ep_overlap.module_fqn "layers.*.moe" \
    --compile.pass_pipeline mutation-functionalization+npu_auto_overlap \
    "$@"
