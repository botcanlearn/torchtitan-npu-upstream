#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run from the repository root; append CLI arguments to override defaults.
# USE_GOLDEN=1 selects reference operators on either A3 or A5.
# For precision comparisons, append --debug.seed 42 --debug.deterministic.

set -euo pipefail

NGPU="${NGPU:-8}"
WORLD_SIZE="${NGPU}"

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4_1}"
CONFIG="${CONFIG:-deepseek_v4_1_flash_40layers_16experts_multimodal}"

# Dataloader & Checkpoint
DATASET="${DATASET:-cc12m-test}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV41_tokenizer}"
CKPT_SAVE_LOAD_PATH="${CKPT_SAVE_LOAD_PATH:-checkpoint}"

# Parallelism
TP=1
PP=1
EP=8
CP=1
DP_SHARD=8
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training (retain the validated eager / AdamW / FullAC resource crop)
SEQ_LEN=512
MBS=1
GBS=8
STEPS=40

USE_GOLDEN="${USE_GOLDEN:-0}"
DEBUG_ARGS="
    --debug.no-moe-force-load-balance
    --debug.print-config
"

DATALOADER_ARGS=(--dataloader.dataset "${DATASET}")
# Leave the path unset so each upstream dataset registration keeps its source.
if [[ -n "${DATASET_PATH:-}" ]]; then
    DATALOADER_ARGS+=(--dataloader.dataset-path "${DATASET_PATH}")
fi

PARALLELISM_ARGS="
    --parallelism.spmd-backend ${SPMD_BACKEND}
    --parallelism.data-parallel-shard-degree ${DP_SHARD}
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE}
    --parallelism.expert-parallel-degree ${EP}
    --parallelism.tensor-parallel-degree ${TP}
    --parallelism.context-parallel-degree ${CP}
    --parallelism.pipeline-parallel-degree ${PP}
    --parallelism.fsdp-reshard-after-forward always
    --parallelism.context-parallel-load-balancer None
    --parallelism.enable-sequence-parallel
"

TRAINING_ARGS="
    --compile.no-enable
    --training.local-batch-size ${MBS}
    --training.global-batch-size ${GBS}
    --training.seq-len ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
    --metrics.log-freq 1
"

LR_SCHEDULER_ARGS="
    --lr-scheduler.warmup-steps 2
    --lr-scheduler.total-steps ${STEPS}
    --lr-scheduler.decay-ratio 1.0
    --lr-scheduler.decay-type cosine
    --lr-scheduler.min-lr-factor 0.01
"

# AdamW keeps its model recipe's param_groups (eps=1e-6). The scalar fields
# below configure the explicit --optimizer.name Muon selection.
OPTIMIZER_ARGS="
    --optimizer.name AdamW
    --optimizer.implementation fused
    --optimizer.lr 1.0e-5
    --optimizer.beta1 0.9
    --optimizer.beta2 0.95
    --optimizer.eps 1.0e-8
    --optimizer.weight-decay 1.0e-1
    --optimizer.muon-momentum 0.95
    --optimizer.muon-enable-nesterov
    --optimizer.muon-ns-steps 10
    --optimizer.muon-adjust-lr-fn match_rms_adamw
"
OPTIMIZER_OVERRIDES="torchtitan_npu.override.common.optimizer.virtual"

CHECKPOINT_ARGS="
    --checkpoint.no-enable
    --checkpoint.load-only
    --checkpoint.interval 10
"

PROFILER_ARGS="
    --profiler.no-enable-profiling
    --profiler.profile-freq 10
    --profiler.profiler-active 10
    --profiler.profiler-warmup 0
"

COMM_ARGS="
    --comm.init-timeout-seconds 7200
    --comm.train-timeout-seconds 600
"

# The text-rope entry is a hardware choice: A3 defaults to asc_complex; the A5
# wrapper supplies asc_partial through the same CLI_OVERRIDES channel.
if [[ "${USE_GOLDEN}" == "1" ]]; then
    NPU_OPS_OVERRIDES=(torchtitan_npu.override.common.rope.workaround)
    DEFAULT_TEXT_ROPE_OVERRIDE=""
else
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rms_norm.asc
        torchtitan_npu.override.common.rope.asc_half_rotation
        torchtitan_npu.override.common.token_dispatcher.asc
        torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post
    )
    DEFAULT_TEXT_ROPE_OVERRIDE="torchtitan_npu.override.common.rope.asc_complex"
fi

# Wrapper defaults extend the same override.imports list, as in DSV4.
CLI_OVERRIDES="${CLI_OVERRIDES:-${DEFAULT_TEXT_ROPE_OVERRIDE}}"

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train.sh \
    --hf-assets-path "${HF_ASSETS_PATH}" \
    "${DATALOADER_ARGS[@]}" \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $LR_SCHEDULER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --checkpoint.folder "${CKPT_SAVE_LOAD_PATH}" \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    $CLI_OVERRIDES \
    "$@"
