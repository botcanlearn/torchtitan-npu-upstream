# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HiFloat8 quantization primitive."""

__all__ = ["quantize_hifloat8"]

import torch
import torch_npu

from torchao_npu.quantization.quant_configs import HiF8QuantizeConfig


def quantize_hifloat8(value: torch.Tensor, config: HiF8QuantizeConfig):
    """Quantize a tensor to HiFloat8 with a float32 scale."""

    if value.dtype not in (torch.float16, torch.bfloat16):
        value = value.to(torch.bfloat16)
    q_data, descale = torch_npu.npu_dynamic_quant(
        value,
        dst_type=config.elem_dtype,
        dst_type_max=config.dst_type_max,
        quant_mode=config.quant_mode,
    )
    return q_data, descale.to(torch.float32)
