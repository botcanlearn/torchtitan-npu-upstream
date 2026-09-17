# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.transform import _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors.block_mx_wrapper_tensor import BlockMXTrainingWeightWrapperTensor

from ..testing_utils import target_devices


def test_block_mx_handler_wraps_parameter_and_preserves_requires_grad():
    parameter = nn.Parameter(torch.randn(2, 4), requires_grad=True)
    config = ParamSwapConfig(
        weight_config=BlockMXQuantizeConfig(),
        activation_config=MXQuantizeConfig(),
    )
    handler = _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[BlockMXQuantizeConfig]

    result = handler(nn.Linear(4, 2), "weight", parameter, (config,))

    assert isinstance(result.data, BlockMXTrainingWeightWrapperTensor)
    assert result.data.weight_config is config.weight_config
    assert result.data.activation_config is config.activation_config
    assert result.requires_grad


def _make_prequantized_wrapper(shape=(64, 128), fsdp_prequantize=True):
    """Build a wrapper carrying pre-quantized FP8 data + scales."""
    weight_config = BlockMXQuantizeConfig(fsdp_prequantize=fsdp_prequantize)
    activation_config = MXQuantizeConfig()
    # torch.randn does not implement a CPU kernel for FP8 dtypes, so build the
    # FP8 tensors by casting from a float source instead.
    b_q = torch.randn(*shape).to(torch.float8_e4m3fn)
    b_s1 = torch.randn(shape[0], 2, 2).to(torch.float8_e8m0fnu)
    b_s2 = torch.randn(2, shape[1], 2).to(torch.float8_e8m0fnu)
    return BlockMXTrainingWeightWrapperTensor._from_prequantized(b_q, b_s1, b_s2, weight_config, activation_config)


def test_prequantized_wrapper_has_prequantized_data():
    wrapper = _make_prequantized_wrapper()
    assert wrapper._has_prequantized_data()
    # Logical dtype stays BF16 while _data holds FP8.
    assert wrapper.dtype == torch.bfloat16
    assert wrapper._data.dtype == torch.float8_e4m3fn


def test_prequantized_wrapper_from_prequantized_preserves_configs():
    weight_config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    activation_config = MXQuantizeConfig()
    b_q = torch.randn(64, 128).to(torch.float8_e4m3fn)
    b_s1 = torch.randn(64, 2, 2).to(torch.float8_e8m0fnu)
    b_s2 = torch.randn(2, 128, 2).to(torch.float8_e8m0fnu)
    wrapper = BlockMXTrainingWeightWrapperTensor._from_prequantized(
        b_q, b_s1, b_s2, weight_config, activation_config, requires_grad=True
    )
    assert wrapper.weight_config is weight_config
    assert wrapper.activation_config is activation_config
    assert wrapper.requires_grad


def test_prequantized_wrapper_transpose_scales():
    """Transposing the last two dims swaps the roles of the two scales."""
    wrapper = _make_prequantized_wrapper(shape=(64, 128))
    b_q, b_s1, b_s2 = (
        wrapper._data,
        wrapper._scale_s1,
        wrapper._scale_s2,
    )
    new_b_q, new_s1, new_s2 = BlockMXTrainingWeightWrapperTensor._transpose_prequantized_scales(b_q, b_s1, b_s2)
    assert new_b_q.shape == (128, 64)
    # new s1 (N-dim) = old K-dim scale B_s2 transposed: (2, 128, 2) -> (128, 2, 2)
    assert new_s1.shape == (128, 2, 2)
    # new s2 (K-dim) = old N-dim scale B_s1 transposed: (64, 2, 2) -> (2, 64, 2)
    assert new_s2.shape == (2, 64, 2)


def test_prequantized_wrapper_reshape_scales_2d_to_3d():
    """A 2D->3D view (adding a leading batch dim) reshapes the scales."""
    wrapper = _make_prequantized_wrapper(shape=(64, 128))
    b_s1, b_s2 = wrapper._scale_s1, wrapper._scale_s2
    new_s1, new_s2 = BlockMXTrainingWeightWrapperTensor._reshape_prequantized_scales(
        b_s1, b_s2, torch.Size([64, 128]), torch.Size([2, 32, 128])
    )
    assert new_s1.shape == (2, 32, 2, 2)
    assert new_s2.shape == (2, 1, 128, 2)


def test_prequantized_wrapper_can_prequantize_falls_back_when_disabled():
    """_can_prequantize returns False when fsdp_prequantize is disabled."""
    wrapper = _make_prequantized_wrapper(fsdp_prequantize=False)
    assert wrapper._can_prequantize(None) is False


def test_prequantized_wrapper_can_prequantize_requires_block_alignment():
    """_can_prequantize requires 32-aligned trailing dims."""
    wrapper = _make_prequantized_wrapper(shape=(64, 128), fsdp_prequantize=True)
    # Non-32-aligned last dim -> cannot prequantize.
    wrapper._data = torch.randn(64, 100).to(torch.float8_e4m3fn)
    assert wrapper._can_prequantize(None) is False


@pytest.mark.parametrize("device", target_devices)
def test_to_inference_weight_matches_the_training_forward(device):
    """The converted single-axis weight must multiply the way the block-MX wrapped one does."""
    weight_config = BlockMXQuantizeConfig()
    activation_config = MXQuantizeConfig()
    A = torch.randn(64, 128, dtype=torch.bfloat16, device=device)  # [M, K]
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)  # [N, K]
    wrapped = BlockMXTrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=activation_config)

    inference_weight = wrapped.to_inference_weight()
    y_training = F.linear(A, wrapped)
    y_inference = F.linear(A, inference_weight)

    assert isinstance(inference_weight, MXTensor), f"got {type(inference_weight).__name__}"
    assert torch.equal(y_training, y_inference), "the inference weight does not reproduce training's forward"


@pytest.mark.parametrize("device", target_devices)
def test_to_inference_weight_with_mxfp4_matches_the_training_forward(device):
    """With mxfp4-QAT the inference weight is FP4, as training's QAT pre-pass produces."""
    weight_config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    )
    activation_config = MXQuantizeConfig()
    A = torch.randn(64, 128, dtype=torch.bfloat16, device=device)  # [M, K]
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)  # [N, K]
    wrapped = BlockMXTrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=activation_config)

    inference_weight = wrapped.to_inference_weight()
    y_training = F.linear(A, wrapped)
    y_inference = F.linear(A, inference_weight)

    assert inference_weight.quant_config.elem_dtype is torch.float4_e2m1fn_x2, (
        f"expected an FP4 inference weight, got {inference_weight.quant_config.elem_dtype}"
    )
    assert torch.equal(y_training, y_inference), "the FP4 inference weight does not reproduce training's forward"
