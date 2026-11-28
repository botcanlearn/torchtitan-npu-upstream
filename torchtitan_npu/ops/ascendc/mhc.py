# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AscendC dispatcher registrations for DeepSeek-V4 mHC."""

import cann_ops_transformer.ops.mhc_post_backward  # noqa: F401
import torch
import torch_npu  # noqa: F401


def _fake_mhc_sinkhorn_backward(grad_y, norm, sum_out, *_):
    return torch.empty_like(grad_y)


def _fake_mhc_pre_backward(
    x,
    phi,
    alpha,
    grad_h_in,
    grad_h_post,
    grad_h_res,
    inv_rms,
    h_mix,
    h_pre,
    h_post,
    gamma=None,
    hc_eps=1e-6,
    grad_x_post=None,
    # Absorb optional args appended by newer op-plugin schemas (e.g.
    # inner_precise, op-plugin#5836): torch_npu's autograd passes every schema
    # argument positionally, and none of them affect the meta outputs.
    *_,
):
    return (
        torch.empty_like(x),
        torch.empty_like(phi),
        torch.empty_like(alpha),
        torch.empty_like(phi[:, 0]),
        torch.empty_like(gamma) if gamma is not None else torch.empty(0, device=phi.device, dtype=phi.dtype),
    )


torch.library.register_fake("npu::npu_mhc_pre_backward")(_fake_mhc_pre_backward)
torch.library.register_fake("npu::npu_mhc_sinkhorn_backward")(_fake_mhc_sinkhorn_backward)


def _mhc_post_setup_context(ctx, inputs, output):
    x, h_res, h_out, h_post = inputs
    ctx.save_for_backward(x, h_res, h_out, h_post)


def _mhc_post_backward(ctx, grad_output):
    x, h_res, h_out, h_post = ctx.saved_tensors
    return torch.ops.cann_ops_transformer.mhc_post_backward(grad_output, x, h_res, h_out, h_post)


torch.library.register_autograd(
    "cann_ops_transformer::mhc_post", _mhc_post_backward, setup_context=_mhc_post_setup_context
)


def _mhc_pre_sinkhorn_setup_context(ctx, inputs, output, keyword_only_inputs=None):
    """Save the operands ``mhc_pre_sinkhorn_backward`` needs.

    torch calls the callback with keyword arguments, spelling the extra
    ``keyword_only_inputs`` only once the schema declares keyword-only
    arguments; both shapes must be accepted and the names must stay exactly as
    torch passes them.  ops-transformer added ``*, bool enable_hf32=True``
    (next to the optional ``pre_mix``) to ``mhc_pre_sinkhorn``; those defaults
    only select optional features, never reach the backward formula, and the
    fused HcPre path never passes ``pre_mix``, so the backward keeps its
    nine-gradient contract.
    """
    x, phi, alpha, bias = inputs[:4]
    ctx.save_for_backward(x, phi, alpha, bias, *output[3:])
    # ``hcEps`` belongs to the stable positional prefix
    # (x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag); the
    # optional tail appended after ``outFlag`` never shifts it.
    ctx.hc_eps = inputs[6]
    ctx.grad_output_meta = tuple((tensor.shape, tensor.dtype, tensor.device) for tensor in output[:3])
    # Avoid materializing gradients for outputs the graph does not consume.
    ctx.set_materialize_grads(False)
    # Saved auxiliary outputs are not differentiable in this backward. A premix
    # operator that needs the h_pre gradient must use its own autograd registration.
    ctx.mark_non_differentiable(*output[3:])


def _mhc_pre_sinkhorn_backward(ctx, grad_h_in, grad_h_post, grad_h_res, *_):
    (
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
    ) = ctx.saved_tensors

    # Match the kernel outputs even when their metadata differs from the inputs.
    grad_h_in, grad_h_post, grad_h_res = (
        torch.zeros(shape, dtype=dtype, device=device) if grad is None else grad
        for grad, (shape, dtype, device) in zip((grad_h_in, grad_h_post, grad_h_res), ctx.grad_output_meta, strict=True)
    )

    grad_x, grad_phi, grad_alpha, grad_bias = torch.ops.cann_ops_transformer.mhc_pre_sinkhorn_backward(
        grad_h_in,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        ctx.hc_eps,
    )
    return grad_x, grad_phi, grad_alpha, grad_bias, None, None, None, None, None


torch.library.register_autograd(
    "cann_ops_transformer::mhc_pre_sinkhorn",
    _mhc_pre_sinkhorn_backward,
    setup_context=_mhc_pre_sinkhorn_setup_context,
)
