#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 adds CPU affinity, partial text RoPE, SwiGLUGroup experts, sparse
# attention and mHC Sinkhorn to A3; USE_GOLDEN=1 keeps the pure reference list.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Use npu-smi info -t topo to check CPU affinity on the target host.
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"

if [[ "${USE_GOLDEN:-0}" != "1" ]]; then
    # asc_partial replaces the A3 text-rope choice (asc_complex); both claim
    # the same public ComplexRoPE.Config, so only one may be active.
    export CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rope.asc_partial \
                                           torchtitan_npu.override.common.swiglu_group.asc \
                                           torchtitan_npu.override.common.swiglu_group.asc_shared_experts \
                                           torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc \
                                           torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn}"
fi

exec bash "${SCRIPT_DIR}/deepseek_v4_1_flash_8p_cpt_4k_a3.sh" "$@"
