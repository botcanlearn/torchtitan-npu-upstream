#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Thin A3 entry for the V4.1 multimodal run.  The *_multimodal_a3 recipe carries the
# full training semantics — including the fused override list — so the
# printed Trainer Config is the single source of truth.  Data paths come in
# through the standard CLI; see examples/deepseek_v4_1/readme.md for the
# full command forms (reference and A5 variants select their recipe through
# CONFIG=).

set -euo pipefail

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4_1}" \
CONFIG="${CONFIG:-deepseek_v4_1_flash_40layers_16experts_multimodal_a3}" \
NGPU="${NGPU:-8}" \
bash scripts/run_train.sh "$@"
