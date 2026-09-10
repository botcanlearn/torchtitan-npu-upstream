# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization configurations, filters, and parameter transforms."""

from types import MappingProxyType

import torch
import torch_npu

# torch dtype -> torch_npu dtype accepted by the NPU quant ops.
_NPU_DTYPE_DICT = MappingProxyType(
    {
        torch.float8_e4m3fn: torch_npu.float8_e4m3fn,
        torch.float8_e5m2: torch_npu.float8_e5m2,
        torch.float8_e8m0fnu: torch_npu.float8_e8m0fnu,
        torch.float4_e2m1fn_x2: torch_npu.float4_e2m1fn_x2,
    }
)
