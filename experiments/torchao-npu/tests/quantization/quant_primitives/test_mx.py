# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch_npu
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantization.quant_primitives.mx import _get_fp4_e2m1_pair_lut, mxfp4_dequantize


def test_fp4_pair_lut_decodes_both_nibble_orders():
    low_first = _get_fp4_e2m1_pair_lut(torch.device("cpu"), torch.float32, True)
    high_first = _get_fp4_e2m1_pair_lut(torch.device("cpu"), torch.float32, False)

    assert low_first.shape == (256, 2)
    assert torch.equal(low_first[0x21], torch.tensor([0.5, 1.0]))
    assert torch.equal(high_first[0x21], torch.tensor([1.0, 0.5]))


def test_mxfp4_dequantize_unpacks_values_and_e8m0_scale():
    data = torch.full((1, 16), 0x21, dtype=torch.uint8)
    scale = torch.full((1, 1, 2), 127, dtype=torch.uint8)

    result = mxfp4_dequantize(
        data,
        scale,
        axis=-1,
        block_size=32,
        output_shape=torch.Size((1, 32)),
        output_dtype=torch.float32,
    )

    expected = torch.tensor([0.5, 1.0] * 16).reshape(1, 32)
    assert torch.equal(result, expected)


def test_mxfp4_dequantize_rejects_partial_blocks():
    with pytest.raises(AssertionError, match="divisible by block_size"):
        mxfp4_dequantize(
            torch.zeros(1, 15, dtype=torch.uint8),
            torch.zeros(1, 1, 2, dtype=torch.uint8),
            axis=-1,
            block_size=32,
            output_shape=torch.Size((1, 30)),
            output_dtype=torch.float32,
        )


# =========================================================================
# Tests for mx_quantize (permutation-aware wrapper over npu_dynamic_mx_quant)
# =========================================================================


@pytest.mark.parametrize(
    "tensor, axis",
    [
        # 1. Dense (is_contiguous() True) -> if branch, no transpose.
        (torch.randn(256, 128, device="npu", dtype=torch.bfloat16), -1),
        # 2. Quant axis strided (stride(axis) != 1): a transposed 2D tensor.
        (torch.randn(128, 256, device="npu", dtype=torch.bfloat16).transpose(0, 1), 1),
        # 3. Pure perm view with innermost-contiguous quant axis (like wo_a).
        (torch.randn(4, 128, 256, device="npu", dtype=torch.bfloat16).transpose(1, 2), 1),
        # 4. Non-pure strided view, stride(axis) == 1 but permute stays non-dense.
        (torch.as_strided(torch.randn(4, 4, device="npu", dtype=torch.bfloat16), (2, 4), (2, 1)), 1),
    ],
    ids=["dense", "strided_axis", "perm_view", "non_pure"],
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mx_quantize_matches_raw_op_for_all_layouts(tensor, axis, elem_dtype):
    """mx_quantize matches npu_dynamic_mx_quant on all four input layouts.

    These are value-equivalence checks: they verify correctness, not that a real
    transpose was avoided (that would require profiling). The point is each of
    the four branch inputs must produce identical y/scale to the raw op.
    """
    from torchao_npu.quantization.quant_primitives.mx import mx_quantize

    def as_uint8(t):
        # FP8 tensors can't be compared element-wise with torch.equal on NPU;
        # compare exact bytes instead. Each value is at most 1 byte (FP8, or FP4
        # packed two per byte), so a uint8 view reinterprets the storage without
        # a copy. contiguous() makes the view legal on permuted (non-contiguous)
        # outputs.
        return t.contiguous().view(torch.uint8)

    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    y, scale = mx_quantize(tensor, axis, config)
    y_ref, scale_ref = torch_npu.npu_dynamic_mx_quant(
        tensor,
        axis=axis,
        dst_type=config.npu_elem_dtype,
        block_size=config.block_size,
        round_mode=config.round_mode,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    assert y.shape == y_ref.shape, f"y shape mismatch: {y.shape} vs {y_ref.shape}"
    assert y.dtype == y_ref.dtype, f"y dtype mismatch: {y.dtype} vs {y_ref.dtype}"
    assert torch.equal(as_uint8(y), as_uint8(y_ref)), "y values mismatch"
    assert scale.shape == scale_ref.shape, f"scale shape mismatch: {scale.shape} vs {scale_ref.shape}"
    assert torch.equal(as_uint8(scale), as_uint8(scale_ref)), "scale values mismatch"


@pytest.mark.parametrize("which", ["left", "right"])
@pytest.mark.parametrize("case_b", ["dense", "strided_axis", "perm_view", "non_pure"])
@pytest.mark.parametrize("case_a", ["dense", "strided_axis", "perm_view", "non_pure"])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2])
def test_mx_quantize_quant_matmul_matches_raw_op(elem_dtype, case_a, case_b, which):
    """npu_quant_matmul output should be identical whether an operand is quantized by
    mx_quantize or by the raw op.
    """
    from torchao_npu.quantization.quant_primitives.mx import mx_quantize

    def matmul_operand(case, batch, rows, cols):
        """Build a 3D ``(batch, rows, cols)`` bf16 batched matmul operand in one
        of four layouts:

        - ``dense``: contiguous.
        - ``strided_axis`` / ``perm_view``: a 2-way transposed view (non-contiguous).
        - ``non_pure``: an as_strided view with a row gap (non-contiguous, in-bounds).
        """
        if case == "strided_axis" or case == "perm_view":
            return torch.randn(batch, cols, rows, device="npu", dtype=torch.bfloat16).transpose(-1, -2)

        elif case == "non_pure":
            base = torch.randn(batch, rows, cols + 1, device="npu", dtype=torch.bfloat16)
            return torch.as_strided(base, (batch, rows, cols), (rows * (cols + 1), cols + 1, 1))

        elif case == "dense":
            return torch.randn(batch, rows, cols, device="npu", dtype=torch.bfloat16)

        else:
            raise ValueError(f"`case` only support {['dense', 'strided_axis', 'perm_view', 'non_pure']}")

    b, m, k, n = 2, 64, 128, 256
    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    left = matmul_operand(case_a, b, m, k)
    right = matmul_operand(case_b, b, k, n)

    def quant_operand(t, axis, use_mx):
        if use_mx:
            return mx_quantize(t, axis, config)

        else:
            return torch_npu.npu_dynamic_mx_quant(
                t,
                axis=axis,
                dst_type=config.npu_elem_dtype,
                block_size=config.block_size,
                round_mode=config.round_mode,
                scale_alg=config.scale_alg,
                dst_type_max=config.dst_type_max,
            )

    def matmul(a_q, a_s, b_q, b_s):
        return torch_npu.npu_quant_matmul(
            a_q,
            b_q,
            b_s,
            pertoken_scale=a_s,
            output_dtype=left.dtype,
            group_sizes=[1, 1, config.block_size],
            scale_dtype=config.npu_scale_dtype,
            pertoken_scale_dtype=config.npu_scale_dtype,
            x1_dtype=config.npu_matmul_dtype,
            x2_dtype=config.npu_matmul_dtype,
        )

    # mx_quantize on the selected operand; the other operand via the raw op.
    a_q, a_s = quant_operand(left, -1, use_mx=(which == "left"))
    b_q, b_s = quant_operand(right, -2, use_mx=(which == "right"))
    # Both-raw reference.
    a_q_r, a_s_r = quant_operand(left, -1, use_mx=False)
    b_q_r, b_s_r = quant_operand(right, -2, use_mx=False)

    y = matmul(a_q, a_s, b_q, b_s)
    y_ref = matmul(a_q_r, a_s_r, b_q_r, b_s_r)

    assert y.shape == y_ref.shape
    assert torch.equal(y, y_ref), "npu_quant_matmul result differs between mx_quantize and raw op"


# =========================================================================
# Tests for mx_quantize_dual_axis (permutation-aware wrapper over
# npu_dynamic_mx_quant_with_dual_axis)
# =========================================================================


@pytest.mark.parametrize(
    "tensor",
    [
        # 1. Dense (is_contiguous() True) -> identity permutation, no swap.
        torch.randn(256, 128, device="npu", dtype=torch.bfloat16),
        # 2. 2D transpose -> the two quant dims land reversed, so swap happens.
        torch.randn(128, 256, device="npu", dtype=torch.bfloat16).t(),
        # 3. Pure permutation view with reordered leading dims (the two quant
        #    dims keep their order) -> leading-only permutation, no swap.
        torch.randn(5, 3, 64, 128, device="npu", dtype=torch.bfloat16).permute(1, 0, 2, 3),
        # 4. Non-pure strided view (strides don't match a permutation) -> no free
        #    permutation.
        torch.as_strided(torch.randn(4, 4, device="npu", dtype=torch.bfloat16), (2, 4), (2, 1)),
        # 5. Pure permutation view that crosses the leading/trailing-two
        #    boundary: restoring density would move the leading dim between the
        #    two quant dims, so the permuted tensor stays non-dense (inevitable
        #    copy), but outputs must still match.
        torch.randn(64, 3, 128, device="npu", dtype=torch.bfloat16).permute(1, 0, 2),
    ],
    ids=["dense", "transposed", "perm_view", "non_pure", "cross_boundary"],
)
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mx_quantize_dual_axis_matches_raw_op_for_all_layouts(tensor, elem_dtype):
    """mx_quantize_dual_axis matches npu_dynamic_mx_quant_with_dual_axis on all layouts.

    These are value-equivalence checks: they verify correctness, not that a real
    transpose was avoided (that would require profiling). The point is each of the
    five branch inputs must produce identical y1/s1/y2/s2 to the raw op.
    """
    from torchao_npu.quantization.quant_primitives.mx import mx_quantize_dual_axis

    def as_uint8(t):
        # FP8 tensors can't be compared element-wise with torch.equal on NPU;
        # compare exact bytes instead. Each value is at most 1 byte, so a uint8
        # view reinterprets the storage without a copy. contiguous() makes the
        # view legal on permuted (non-contiguous) outputs.
        return t.contiguous().view(torch.uint8)

    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    y1, s1, y2, s2 = mx_quantize_dual_axis(tensor, config)
    r1, rs1, r2, rs2 = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        tensor,
        round_mode=config.round_mode,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )

    assert y1.shape == r1.shape, f"y1 shape mismatch: {y1.shape} vs {r1.shape}"
    assert y2.shape == r2.shape, f"y2 shape mismatch: {y2.shape} vs {r2.shape}"
    assert s1.shape == rs1.shape, f"s1 shape mismatch: {s1.shape} vs {rs1.shape}"
    assert s2.shape == rs2.shape, f"s2 shape mismatch: {s2.shape} vs {rs2.shape}"
    assert torch.equal(as_uint8(y1), as_uint8(r1)), "y1 values mismatch"
    assert torch.equal(as_uint8(y2), as_uint8(r2)), "y2 values mismatch"
    assert torch.equal(as_uint8(s1), as_uint8(rs1)), "s1 values mismatch"
    assert torch.equal(as_uint8(s2), as_uint8(rs2)), "s2 values mismatch"


@pytest.mark.parametrize("which", ["left", "right"])
@pytest.mark.parametrize("case_b", ["dense", "transposed", "perm_view", "non_pure"])
@pytest.mark.parametrize("case_a", ["dense", "transposed", "perm_view", "non_pure"])
@pytest.mark.parametrize("elem_dtype", [torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2])
def test_mx_quantize_dual_axis_quant_matmul_matches_raw_op(elem_dtype, case_a, case_b, which):
    """npu_quant_matmul output should be identical whether an operand is dual-axis
    quantized by mx_quantize_dual_axis or by the raw op.
    """
    from torchao_npu.quantization.quant_primitives.mx import mx_quantize_dual_axis

    def matmul_operand(case, batch, rows, cols):
        """Build a 3D ``(batch, rows, cols)`` bf16 operand in one of four layouts."""
        if case == "transposed" or case == "perm_view":
            return torch.randn(batch, cols, rows, device="npu", dtype=torch.bfloat16).transpose(-1, -2)
        elif case == "non_pure":
            base = torch.randn(batch, rows, cols + 1, device="npu", dtype=torch.bfloat16)
            return torch.as_strided(base, (batch, rows, cols), (rows * (cols + 1), cols + 1, 1))
        elif case == "dense":
            return torch.randn(batch, rows, cols, device="npu", dtype=torch.bfloat16)
        else:
            raise ValueError(f"`case` only support {['dense', 'transposed', 'perm_view', 'non_pure']}")

    b, m, k, n = 2, 64, 128, 256
    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    left = matmul_operand(case_a, b, m, k)
    right = matmul_operand(case_b, b, k, n)

    def quant_operand(t, use_mx):
        if use_mx:
            return mx_quantize_dual_axis(t, config)
        else:
            return torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                t,
                round_mode=config.round_mode,
                dst_type=config.npu_elem_dtype,
                scale_alg=config.scale_alg,
                dst_type_max=config.dst_type_max,
            )

    def matmul(a_q1, a_s1, b_q2, b_s2):
        # Replicates _MXQuantBMM.forward: x1 = A's k-dim quant (a_q1/a_s1),
        # x2 = b's k-dim quant (b_q2/b_s2).
        return torch_npu.npu_quant_matmul(
            a_q1,
            b_q2,
            b_s2,
            pertoken_scale=a_s1,
            output_dtype=left.dtype,
            group_sizes=[1, 1, config.block_size],
            scale_dtype=config.npu_scale_dtype,
            pertoken_scale_dtype=config.npu_scale_dtype,
            x1_dtype=config.npu_matmul_dtype,
            x2_dtype=config.npu_matmul_dtype,
        )

    # mx_quantize_dual_axis on the selected operand; the other operand via the raw op.
    a_q1, a_s1, _, _ = quant_operand(left, use_mx=(which == "left"))
    _, _, b_q2, b_s2 = quant_operand(right, use_mx=(which == "right"))
    # Both-raw reference.
    a_q1_r, a_s1_r, _, _ = quant_operand(left, use_mx=False)
    _, _, b_q2_r, b_s2_r = quant_operand(right, use_mx=False)

    y = matmul(a_q1, a_s1, b_q2, b_s2)
    y_ref = matmul(a_q1_r, a_s1_r, b_q2_r, b_s2_r)

    assert y.shape == y_ref.shape
    assert torch.equal(y, y_ref), "npu_quant_matmul result differs between mx_quantize_dual_axis and raw op"
