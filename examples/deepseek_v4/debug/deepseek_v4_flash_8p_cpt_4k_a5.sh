#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 wrapper around the common DeepSeek-V4 Flash CPT launcher. Keep the shared
# model, training, and parallelism defaults in the A3 script; this entrypoint
# only supplies A5-specific runtime, quantization, and recompute settings.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5.sh --training.steps 5

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Use the command line `npu-smi info -t topo` to query the CPU affinity of the
# NPU cards for configuration.
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"

# A5 defaults to Inductor; partial RoPE fuses through the asc_partial override.
export CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rope.asc_partial}"

# A5 defaults to block-FP8 quantized training. Append
# --extension.quantization.no-enable-quantized-training for BF16 training.
QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
    --extension.quantization.fsdp-prequantize
    --extension.quantization.li-quantization fp8
)

# Tyro treats activation-checkpoint:selective as a subcommand, so it must be
# the final token after all regular options and override targets.
exec bash "${SCRIPT_DIR}/deepseek_v4_flash_8p_cpt_4k_a3.sh" \
    "${QUANTIZATION_ARGS[@]}" \
    --debug.moe-force-load-balance \
    "$@" \
    activation-checkpoint:selective
