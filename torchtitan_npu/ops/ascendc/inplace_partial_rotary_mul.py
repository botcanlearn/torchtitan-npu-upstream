# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapt CANN partial-RoPE backward for unbacked dynamic shapes."""

import torch

# pyrefly: ignore [missing-import]
from cann_ops_transformer.ops.inplace_partial_rotary_mul.inplace_partial_rotary_mul import (
    InplacePartialRotaryMulFn as _NativeInplacePartialRotaryMulFn,
)


class _InplacePartialRotaryMulFn(
    _NativeInplacePartialRotaryMulFn,  # pyrefly: ignore [invalid-inheritance]
):
    @staticmethod
    def backward(ctx, grad_output):
        r1, r2 = ctx.saved_tensors
        grad_input = grad_output.clone(memory_format=torch.contiguous_format)
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul_backward(
            grad_input,
            r1,
            r2,
            rotary_mode=ctx.rotary_mode,
            partial_slice=ctx.partial_slice,
        )
        return grad_input, None, None, None, None


def inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    *,
    rotary_mode: str = "interleave",
    partial_slice: list[int] | None = None,
) -> None:
    partial_slice = [0, 0] if partial_slice is None else partial_slice
    _InplacePartialRotaryMulFn.apply(x, r1, r2, rotary_mode, partial_slice)
