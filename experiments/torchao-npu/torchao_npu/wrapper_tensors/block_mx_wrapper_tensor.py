# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch import nn
from torchao.prototype.moe_training.utils import unwrap_weight

from torchao_npu.ops.block_mx_ops import (
    to_block_mx_then_bmm,
    to_block_mx_then_grouped_mm,
    to_block_mx_then_mm,
)
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantization.transform import register_parameter_swap_handler
from torchao_npu.wrapper_tensors.base_wrapper_tensor import BaseTrainingWeightWrapperTensor


class BlockMXTrainingWeightWrapperTensor(BaseTrainingWeightWrapperTensor):
    """Applies block MX quantized matmul on NPU.

    Performs a **real** block MX matmul using
    ``to_block_mx_then_mm`` / ``to_block_mx_then_grouped_mm``.

    When ``weight_config.mxfp4_fake_quantize_config`` is set, weights are
    pre-quantized to MXFP4 before the block MX matmul.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        weight_config: BlockMXQuantizeConfig | None = None,
        activation_config: MXQuantizeConfig | None = None,
    ):
        if weight_config is None:
            raise ValueError(f"`weight_config` is required for {type(self).__name__}.")

        if activation_config is None:
            raise ValueError(f"`activation_config` is required for {type(self).__name__}.")

        if type(weight_config) is not BlockMXQuantizeConfig:
            raise ValueError(f"Only `BlockMXQuantizeConfig` is supported for `weight_config` in {type(self).__name__}.")

        if type(activation_config) is not MXQuantizeConfig:
            raise ValueError(f"Only `MXQuantizeConfig` is supported for `activation_config` in {type(self).__name__}.")

        super().__init__(tensor, weight_config=weight_config, activation_config=activation_config)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in (torch.mm, torch.matmul):
            # 2D matmul: A @ B where B is the wrapped weight [K, N]
            return cls._block_mx_mm(args, kwargs)

        elif func is torch._grouped_mm:
            # Grouped matmul: A @ B[E,K,N] with offs=group_list
            return cls._block_mx_grouped_mm(args, kwargs)

        elif func is torch.nn.functional.linear:
            # Linear: A @ weight.T + bias, weight is [N, K]
            return cls._block_mx_linear(args, kwargs)

        elif func is torch.addmm:
            # addmm: bias + A @ B where B is the wrapped weight
            return cls._block_mx_addmm(args, kwargs)

        elif func is torch.bmm:
            # bmm: A @ B where both are 3D and B is the wrapped weight
            return cls._block_mx_bmm(args, kwargs)

        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

    # ------------------------------------------------------------------
    # Per-op helpers
    # ------------------------------------------------------------------

    @classmethod
    def _block_mx_mm(cls, args, kwargs):
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        B_data = unwrap_weight(B)

        with torch._C.DisableTorchFunctionSubclass():
            return to_block_mx_then_mm(A, B_data, B.activation_config, B.weight_config)  # type: ignore

    @classmethod
    def _block_mx_grouped_mm(cls, args, kwargs):
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        group_list = args[2] if len(args) > 2 else kwargs.get("offs")
        B_data = unwrap_weight(B)

        with torch._C.DisableTorchFunctionSubclass():
            return to_block_mx_then_grouped_mm(A, B_data, group_list, B.activation_config, B.weight_config)  # type: ignore

    @classmethod
    def _block_mx_linear(cls, args, kwargs):
        # F.linear(A, weight, bias) — weight is [N, K], A @ weight.T + bias
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        B_data = unwrap_weight(B)
        bias = args[2] if len(args) > 2 else kwargs.get("bias")

        with torch._C.DisableTorchFunctionSubclass():
            result = to_block_mx_then_mm(A, B_data.T, B.activation_config, B.weight_config)  # type: ignore
            if bias is not None:
                result = result + bias
            return result

    @classmethod
    def _block_mx_addmm(cls, args, kwargs):
        # addmm(bias, A, B) — bias + A @ B
        bias, A, B = args[0], args[1], args[2]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        B_data = unwrap_weight(B)

        with torch._C.DisableTorchFunctionSubclass():
            result = to_block_mx_then_mm(A, B_data, B.activation_config, B.weight_config)  # type: ignore
            result = result + bias
            return result

    @classmethod
    def _block_mx_bmm(cls, args, kwargs):
        # bmm(A, B) — both 3D batched matmul, B is the wrapped weight
        A, B = args[0], args[1]
        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        B_data = unwrap_weight(B)

        with torch._C.DisableTorchFunctionSubclass():
            return to_block_mx_then_bmm(A, B_data, B.activation_config, B.weight_config)  # type: ignore


@register_parameter_swap_handler(BlockMXQuantizeConfig)
def _(
    module: nn.Module,
    param_fqn: str,
    param: nn.Parameter,
    extra_args: tuple[Any, ...] = (),
):
    from torchao_npu.configs import ParamSwapConfig

    config: ParamSwapConfig = extra_args[0]

    if not isinstance(config, ParamSwapConfig):
        raise ValueError(f"extra_args[0] must be a ParamSwapConfig, got {type(config).__name__}.")

    if config.activation_config is not None and type(config.activation_config) is not MXQuantizeConfig:
        raise ValueError(
            f"activation_config must be {MXQuantizeConfig.__name__}, got {type(config.activation_config).__name__}."
        )

    if config.weight_config is not None and type(config.weight_config) is not BlockMXQuantizeConfig:
        raise ValueError(
            f"weight_config must be {BlockMXQuantizeConfig.__name__}, got {type(config.weight_config).__name__}."
        )

    return nn.Parameter(
        data=BlockMXTrainingWeightWrapperTensor(
            param.data,
            activation_config=config.activation_config,
            weight_config=config.weight_config,
        ),
        requires_grad=param.requires_grad,
    )


# Safe-unpickling allowlist: DCP loads checkpoints with torch.load(weights_only=True).
torch.serialization.add_safe_globals([BlockMXTrainingWeightWrapperTensor])
