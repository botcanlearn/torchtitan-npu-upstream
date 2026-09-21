#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5.sh \
    --training.enable-cpu-offload \
    "$@"
