# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP4 low-level primitives for NPU.

These are pure tensor helpers (no autograd, no matmul) used by higher-level
ops in :mod:`torchao_npu.ops.mx_ops`
(e.g., ``MXFP4FakeQuantize.forward`` calls :func:`mxfp4_dequantize`).
"""

import functools

import torch
import torch_npu

from torchao_npu import normalize_dim
from torchao_npu.quantization.quant_configs import MXQuantizeConfig


def mx_quantize(
    tensor: torch.Tensor,
    axis: int,
    config: MXQuantizeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MX quantize ``tensor`` along ``axis``, avoiding a real transpose when possible.

    ``npu_dynamic_mx_quant`` requires dense (``is_contiguous()``) input and,
    given a non-dense tensor, silently inserts a ``Contiguous/Transpose`` copy
    before quantizing -- even when the quant axis is already contiguous in
    memory (the op checks ``is_contiguous()`` on the whole tensor, not per-axis).
    This is an inherent limitation of the underlying operator; the kernel has no
    stride handling, so it can only consume a dense layout.

    How a real transpose can be avoided: if the quant axis is the innermost-contiguous
    dimension (``stride(axis) == 1``) and ``tensor`` is not already dense, it is
    a pure permutation view. We then reorder dims so the quant axis lands at -1
    (a view, no copy), quantize along -1 (now dense, so no transpose fires), and
    permute both outputs back to the original layout.

    When a real transpose is inevitable, no view can make the tensor dense and
    the raw op performs a real copy internally. (We still permute the outputs
    back, so the result is correct regardless.) This happens when any of:

    - the quant axis is strided (``stride(axis) != 1``);
    - the tensor is not a pure permutation view, so the permuted tensor stays
      non-dense.

    A dense input is nevertheless quantized as-is: ``npu_dynamic_mx_quant``
    only inserts the transposing copy for a non-dense tensor.

    Warning:
        The transpose-avoiding optimization assumes every dim has a positive
        stride; it does not consider tensors with a stride-0 dim (e.g. a
        broadcast dim) yet.

    Args:
        tensor: Tensor to quantize, shape ``(..., K, ...)``.
        axis: Dimension to quantize along.
        config: MX quantization parameters.

    Returns:
        ``(y, scale)``; ``y`` has the same shape/dtype as ``tensor`` and
        ``scale`` the same shape ``npu_dynamic_mx_quant`` would produce.
    """

    axis = normalize_dim(axis, tensor.ndim)

    # Call the raw op directly
    # 1) tensor.stride(axis) != 1:
    #       a quant axis that isn't innermost-contiguous, a real transpose in inevitable.
    # 2) tensor.stride(axis) == 1 and tensor.is_contiguous():
    #       dense input, ``npu_dynamic_mx_quant`` does not insert a ``Contiguous/Transpose`` copy.
    #
    # The two conditions combined are simplied as below.
    if tensor.stride(axis) != 1 or tensor.is_contiguous():
        y, scale = torch_npu.npu_dynamic_mx_quant(
            tensor,
            axis=axis,
            dst_type=config.npu_elem_dtype,
            block_size=config.block_size,
            round_mode=config.round_mode,
            scale_alg=config.scale_alg,
            dst_type_max=config.dst_type_max,
        )
        return y.view(config.elem_dtype), scale

    else:
        # Non-dense, innermost-contiguous quant axis: permute it to -1.
        # ``stride(axis) == 1`` here, so ordering the other dims by descending
        # stride and appending ``axis`` reproduces the dense layout (a pure view)
        # whenever the tensor is a pure permutation view. Appending ``axis``
        # explicitly -- instead of relying on a descending-stride sort to place
        # it last -- keeps the quant axis at -1 even when another dim ties on
        # stride 1, e.g. a size-1 dim.
        perm_indices = sorted(
            (d for d in range(tensor.ndim) if d != axis),
            key=lambda d: tensor.stride(d),
            reverse=True,
        )
        perm_indices.append(axis)
        tensor_p = tensor.permute(perm_indices)

        y_p, scale_p = torch_npu.npu_dynamic_mx_quant(
            tensor_p,
            axis=-1,
            dst_type=config.npu_elem_dtype,
            block_size=config.block_size,
            round_mode=config.round_mode,
            scale_alg=config.scale_alg,
            dst_type_max=config.dst_type_max,
        )
        y_p = y_p.view(config.elem_dtype)

        # Inverse permutation: y_p dim j corresponds to tensor dim perm_indices[j].
        perm_back_indices = [0] * tensor.ndim
        for j, p in enumerate(perm_indices):
            perm_back_indices[p] = j
        y = y_p.permute(perm_back_indices)

        # scale_p has tensor.ndim+1 dims: the non-axis dims (in permuted order),
        # then the block dim, then the trailing pack-2 dim. The inverse
        # permutation already maps the quant axis to the block slot, so just
        # append tensor.ndim to restore the canonical layout, [.. axis .., block, .. , 2].
        perm_back_indices.append(tensor.ndim)
        scale = scale_p.permute(perm_back_indices)

        return y, scale


def mx_quantize_dual_axis(
    tensor: torch.Tensor,
    config: MXQuantizeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MX quantize ``tensor`` along its two trailing dims, avoiding real transposes when possible.

    ``npu_dynamic_mx_quant_with_dual_axis`` quantizes the two trailing dims of a
    dense tensor and returns ``(y1, s1, y2, s2)`` (``y1``/``s1`` along dim -1,
    ``y2``/``s2`` along dim -2). Given a non-dense tensor it silently inserts a
    ``Contiguous/Transpose`` copy before quantizing, because the kernel has no
    stride handling and can only consume a dense layout. This is an inherent
    limitation of the underlying operator.

    How a real transpose can be avoided: the dims fall into two parts -- the two
    quant dims (-2, -1) and the leading dims. Crossing a dim between parts would
    quantize a different pair and change the semantics, so we only permute within
    each part. We permute each part by descending stride and quantize the result;
    this reproduces the raw op exactly (the op only ever quantizes the trailing
    two dims). When the two quant dims are the two innermost dims the permutation
    is dense, so no copy is needed; otherwise the op copies internally, no worse
    than calling it directly. If the op quantized the two dims in the opposite
    order, we swap ``q1``/``q2`` so the result matches the raw op.

    Args:
        tensor: Tensor to quantize, shape ``(..., row, col)`` (``ndim >= 2``).
        config: MX quantization parameters.

    Returns:
        ``(y1, s1, y2, s2)`` with the same shapes ``npu_dynamic_mx_quant_with_dual_axis``
        would produce for ``tensor``.
    """
    # Permute each part so that, within each part, the strides run descending.
    strides = tensor.stride()
    # Part 1 (leading dims): order them by stride so they run outermost-first.
    lead = sorted(range(tensor.ndim - 2), key=lambda d: strides[d], reverse=True)
    # Part 2 (the two quant dims): order them descending so the innermost one
    # lands at -1. Whether this reverses the original [-2, -1] decides the swap.
    swap = strides[tensor.ndim - 1] > strides[tensor.ndim - 2]
    quant_order = [tensor.ndim - 2, tensor.ndim - 1] if not swap else [tensor.ndim - 1, tensor.ndim - 2]
    perm_indices = [*lead, *quant_order]
    tensor_p = tensor.permute(perm_indices)

    y1_p, s1_p, y2_p, s2_p = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        tensor_p,
        round_mode=config.round_mode,
        dst_type=config.npu_elem_dtype,
        scale_alg=config.scale_alg,
        dst_type_max=config.dst_type_max,
    )
    y1_p = y1_p.view(config.elem_dtype)
    y2_p = y2_p.view(config.elem_dtype)

    # Inverse permutation of the tensor dims. Applying it to any op output --
    # appending the extra block/pack dim for a scale -- returns the tensor to the
    # original layout.
    perm_back_indices = [0] * tensor.ndim
    for j, p in enumerate(perm_indices):
        perm_back_indices[p] = j

    y1 = y1_p.permute(perm_back_indices)
    y2 = y2_p.permute(perm_back_indices)
    s1 = s1_p.permute([*perm_back_indices, tensor.ndim])
    s2 = s2_p.permute([*perm_back_indices, tensor.ndim])

    # The op's q1 is along tensor_p's -1. When the two quant dims were permuted
    # in the opposite order, that dim is the original -2: swap the outputs so q1
    # tracks the original -1 (and q2 the original -2).
    if swap:
        y1, y2 = y2, y1
        s1, s2 = s2, s1
    return y1, s1, y2, s2


def mxfp4_dequantize(
    data: torch.Tensor,
    scale: torch.Tensor,
    axis: int,
    block_size: int,
    output_shape: torch.Size,
    output_dtype: torch.dtype,
    low_first: bool = True,
) -> torch.Tensor:
    """Dequantize MXFP4 data from ``torch_npu.npu_dynamic_mx_quant`` output.

    Args:
        data: uint8 tensor (output y from npu_dynamic_mx_quant).
              Last dim is halved (2 FP4 values packed per byte).
        scale: uint8 E8M0 tensor (output mxscale_out).
               ndim = data.ndim + 1, with a trailing 2 dim packing scale pairs.
        axis: Quantization axis used in npu_dynamic_mx_quant.
        block_size: Block size used in npu_dynamic_mx_quant.
        output_dtype: Target dtype for the dequantized output.
        low_first: If True, low nibble is the first element in each packed byte.
    """
    assert output_shape[axis] % block_size == 0, f"quant dim must be divisible by block_size ({block_size})"

    # Use cached 256-entry LUT (indexed by full byte, no nibble splitting)
    # pyrefly: ignore [bad-assignment, bad-argument-type, bad-argument-count]
    # (_get_fp4_e2m1_pair_lut is wrapped by @torch.compiler.disable, which
    # collapses its signature to (fn) -> Any in pyrefly's view.)
    lut: torch.Tensor = _get_fp4_e2m1_pair_lut(data.device, torch.bfloat16, low_first)
    idx = data.to(torch.uint8).reshape(-1).to(torch.long)
    values = torch.index_select(lut, dim=0, index=idx)

    # Reconstruct original input shape (last dim doubled after unpacking)
    values = values.reshape(*data.shape[:-1], data.shape[-1] * 2)
    if data.shape[-1] * 2 != output_shape[-1]:
        values = values.narrow(-1, 0, output_shape[-1])

    orig_shape = values.shape
    orig_ndim = values.ndim
    pos_axis = axis if axis >= 0 else axis + orig_ndim
    qdim = orig_shape[pos_axis]
    num_blocks = qdim // block_size

    # Unpack scale: NPU packed format → [..., num_blocks, ...]
    # The trailing 2 dim stores scale pairs; move it next to the packed-block dim
    scale = scale.to(torch.uint8)
    scale = scale.movedim(-1, pos_axis + 1)
    scale = scale.flatten(pos_axis, pos_axis + 1)  # [..., packed_blocks*2, ...]
    scale = scale.narrow(pos_axis, 0, num_blocks)  # trim padding when num_blocks is odd

    # E8M0 uint8 → bf16 scale: 2^(e - 127)
    scale = torch.exp2(scale.to(torch.bfloat16) - 127.0)

    # Block-wise broadcast multiply
    # Values: [..., qdim, ...] → [..., num_blocks, block_size, ...]
    values = values.unflatten(pos_axis, (num_blocks, block_size))

    # Scale: [..., num_blocks, ...] → [..., num_blocks, 1, ...]
    scale = scale.unsqueeze(pos_axis + 1)

    result = values * scale  # broadcasts over block_size
    result = result.reshape(*orig_shape)

    return result.to(output_dtype)


@torch.compiler.disable
@functools.cache
def _get_fp4_e2m1_pair_lut(device, dtype=torch.bfloat16, low_first: bool = True) -> torch.Tensor:
    """LUT mapping a packed uint8 byte to a pair of decoded FP4 values.

    Returns shape [256, 2], indexed by the full byte value.
    low nibble → index 0, high nibble → index 1 (or swapped if low_first=False).
    """
    fp4_vals = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
        dtype=dtype,
    )
    p = torch.arange(256, device=device)
    # Each byte packs two FP4 values: low nibble (bits 0-3) and high nibble (bits 4-7).
    # Index fp4_vals (16 entries) by these nibbles.
    low_nibble = p & 0x0F  # mask out high nibble → 0..15
    high_nibble = p >> 4  # shift down high nibble → 0..15
    idx0, idx1 = (low_nibble, high_nibble) if low_first else (high_nibble, low_nibble)
    return torch.stack([fp4_vals[idx0], fp4_vals[idx1]], dim=1)
