#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Thin A5 entry for the V4.1 multimodal run.  The *_multimodal_a5 recipe carries the
# full training semantics — the A3 fused stack plus the A5-only sparse
# attention and mHC Sinkhorn kernels.  CPU affinity defaults to the
# configured eight-device CPU mapping and can be overridden through
# CPU_AFFINITY_CONF; data paths come in through the standard CLI.

set -euo pipefail

# Use npu-smi info -t topo to check CPU affinity on the target host.
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4_1}" \
CONFIG="${CONFIG:-deepseek_v4_1_flash_40layers_16experts_multimodal_a5}" \
NGPU="${NGPU:-8}" \
bash scripts/run_train.sh "$@"
