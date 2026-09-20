# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module-swap configuration for the V4.1 sparse attention quantization path."""

__all__ = ["QuantV41SparseAttentionConfig"]

from dataclasses import dataclass
from types import MethodType

from torchao.quantization.qat import QATStep
from torchao.quantization.transform_module import register_quantize_module_handler

from torchao_npu.configs.module_swap import ModuleSwapConfig


@dataclass(kw_only=True, slots=True)
class QuantV41SparseAttentionConfig(ModuleSwapConfig):
    """Install mixed quant sparse attention with FP8 SWA_KV / FP4 CMP_KV and BF16 scales."""


@register_quantize_module_handler(QuantV41SparseAttentionConfig)
def _quant_v41_sparse_attention_transform(module, config: QuantV41SparseAttentionConfig):
    from torchao_npu.quantized_modules.v41_sparse_attention import QuantV41SparseAttention

    # Keep the host forward's validation and distillation loss wiring intact.
    if not hasattr(module, "_torchao_npu_original_compute_attention"):
        module._torchao_npu_original_compute_attention = module._compute_attention
    if config.step == QATStep.PREPARE:
        module._torchao_npu_module_swap_config = config
        module._compute_attention = MethodType(QuantV41SparseAttention.forward, module)
    else:
        module._compute_attention = module._torchao_npu_original_compute_attention
    return module
