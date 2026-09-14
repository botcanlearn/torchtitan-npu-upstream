# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Concrete module-swap quantization configs."""

__all__ = ["QuantLightningIndexerConfig", "quant_mode_for_config"]

from torchao_npu.configs.module_swap_configs.quant_lightning_indexer import (
    QuantLightningIndexerConfig,
    quant_mode_for_config,
)
