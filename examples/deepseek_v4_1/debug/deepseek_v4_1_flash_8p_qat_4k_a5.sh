#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 wrapper around the common DeepSeek-V4.1 Flash CPT launcher. Keep the shared
# model, training, and parallelism defaults in the A3 script; this entrypoint
# only supplies A5-specific runtime and fused-operator settings.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_qat_4k_a5.sh --training.steps 5

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
    --extension.quantization.enable-mxfp4-qat
    --extension.quantization.enable-sparse-attention-quantization
)

exec bash "${SCRIPT_DIR}/deepseek_v4_1_flash_8p_cpt_4k_a5.sh" \
    "${QUANTIZATION_ARGS[@]}" \
    "$@" \
