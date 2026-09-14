# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public quantization configurations for torchao-npu."""

from torchao_npu.configs.module_swap import ModuleSwapConfig
from torchao_npu.configs.module_swap_configs import QuantLightningIndexerConfig, quant_mode_for_config
from torchao_npu.configs.param_swap import ParamSwapConfig

__all__ = [
    "ModuleSwapConfig",
    "ParamSwapConfig",
    "QuantLightningIndexerConfig",
    "quant_mode_for_config",
]
