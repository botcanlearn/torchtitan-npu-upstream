#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Two nodes, 8 NPUs per node, EP16 / DP-shard16.
# Launch simultaneously on both nodes; NODE_IPS is ordered [master, worker].
#   NODE_IPS=192.168.0.30,192.168.0.107 NGPU=8 STEPS=5 COMPILE_ENABLE=0 \
#     bash examples/deepseek_v4/debug/deepseek_v4_flash_16p_cpt_4k_a3.sh
# USE_GOLDEN=1 selects Golden. For deterministic execution, append
# --debug.seed 42 --debug.deterministic to the command line.

set -euo pipefail

NGPU="${NGPU:-8}"
NODE_IPS="${NODE_IPS:?Set NODE_IPS to the two execution-node IPs, master first}"
NNODES=$(awk -F, '{print NF}' <<< "${NODE_IPS}")
if [[ "${NGPU}" != "8" || "${NNODES}" != "2" ]]; then
    echo "16P test requires NGPU=8 per node and exactly 2 NODE_IPS" >&2
    exit 2
fi
WORLD_SIZE=$((NGPU * NNODES))

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}"
CONFIG="${CONFIG:-deepseek_v4_flash_43layers_16experts}"

# Dataloader & Checkpoint
DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-tests/assets/c4_test}" # your data path
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV4_tokenizer}" # your tokenizer path
CKPT_SAVE_LOAD_PATH="${CKPT_SAVE_LOAD_PATH:-/path/to/save_ckpt}" # your model save/load ckpt path

# Parallelism
TP=1
PP=1
EP=16
CP=1
DP_SHARD=16
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training
SEQ_LEN=4096
MBS=1
GBS=128
STEPS="${STEPS:-100}"

# Debug
USE_GOLDEN="${USE_GOLDEN:-0}"
DEBUG_ARGS="
    --debug.no-moe-force-load-balance
    --debug.print-config
"

# HF assets
HF_ASSETS_ARGS="
    --hf-assets-path ${HF_ASSETS_PATH}
"

# Dataloader
DATALOADER_ARGS="
    --dataloader.dataset ${DATASET}
    --dataloader.dataset-path ${DATASET_PATH}
"

# Parallelism
PARALLELISM_ARGS="
    --parallelism.spmd-backend ${SPMD_BACKEND}
    --parallelism.data-parallel-shard-degree ${DP_SHARD}
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE}
    --parallelism.expert-parallel-degree ${EP}
    --parallelism.tensor-parallel-degree ${TP}
    --parallelism.context-parallel-degree ${CP}
    --parallelism.pipeline-parallel-degree ${PP}
"

# Compile: preserve the example default; CI smoke may use COMPILE_ENABLE=0.
COMPILE_ENABLE="${COMPILE_ENABLE:-1}"
case "$COMPILE_ENABLE" in
    1) COMPILE_ARGS="--compile.enable --compile.components model --compile.backend inductor" ;;
    0) COMPILE_ARGS="--compile.no-enable" ;;
    *) echo "COMPILE_ENABLE must be 0 or 1" >&2; exit 2 ;;
esac

# Training
TRAINING_ARGS="
    --training.local-batch-size ${MBS}
    --training.global-batch-size ${GBS}
    --training.seq-len ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
"

# Checkpoint
# `checkpoint.folder` is the save/load root (`CKPT_SAVE_LOAD_PATH`). If it
# already contains a valid step-* checkpoint, upstream TorchTitan resumes from
# it; use a new/empty folder when starting a fresh run.
CHECKPOINT_ARGS="
    --checkpoint.no-enable
    --checkpoint.load-only
    --checkpoint.folder ${CKPT_SAVE_LOAD_PATH}
"

# Profiler
PROFILER_ARGS="
    --profiler.no-enable-profiling
    --profiler.save-traces-folder profiling_path
    --profiler.extension.no-enable-online-parse
    --profiler.extension.profiler-start 6
    --profiler.extension.profiler-end 7
    --profiler.extension.profile-ranks 0
"

# Communication
COMM_ARGS="
    --comm.init-timeout-seconds 600
    --comm.train-timeout-seconds 600
"

# 16P Eager smoke uses AdamW: DistMuon bucket-plan validation fails for EP16.
# Muon can be separately tested after the optimizer implementation is fixed.
OPTIMIZER_ARGS="
    --optimizer.name AdamW
    --optimizer.lr 1.0e-5
    --optimizer.beta1 0.9
    --optimizer.beta2 0.95
    --optimizer.eps 1.0e-8
    --optimizer.weight_decay 1.0e-1
    --optimizer.muon_momentum 0.95
    --optimizer.muon_enable_nesterov
    --optimizer.muon_ns_steps 10
    --optimizer.muon_adjust_lr_fn match_rms_adamw
"
OPTIMIZER_OVERRIDES=""

if [[ "${USE_GOLDEN}" == "1" ]]; then
    DEFAULT_CLI_OVERRIDES=""
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rope.workaround
        torchtitan_npu.override.deepseek_v4.sparse_attn.golden
    )
else
    DEFAULT_CLI_OVERRIDES="torchtitan_npu.override.common.rope.asc_complex"
    NPU_OPS_OVERRIDES=(
        # Attention / DSA
        torchtitan_npu.override.common.rms_norm.asc

        torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata
        torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li

        torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata
        torchtitan_npu.override.deepseek_v4.sparse_attn.asc
        # MHC
        torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre
        torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post
        # MoE token dispatcher
        torchtitan_npu.override.common.token_dispatcher.asc
    )
fi

# Wrapper defaults extend override.imports after the base targets.
CLI_OVERRIDES="${CLI_OVERRIDES:-${DEFAULT_CLI_OVERRIDES}}"

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NODE_IPS="${NODE_IPS}" \
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train_multinodes.sh \
    $COMPILE_ARGS \
    $HF_ASSETS_ARGS \
    $DATALOADER_ARGS \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    $CLI_OVERRIDES \
    "$@"
