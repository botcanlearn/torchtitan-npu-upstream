# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AscendC dispatcher registrations for DeepSeek-V4 / V4.1 mHC."""

import cann_ops_transformer.ops.mhc_post_backward  # noqa: F401
import torch
import torch_npu


def allow_hf32() -> bool:
    """The run-wide ACLNN HF32 switch every fused mHC path follows.

    ``True`` matches the packages that predate the switch, mirroring the
    fallback ops-transformer itself applies.  The forward dispatch reads the
    same attribute internally; the backward formulas receive it explicitly so
    both directions round the same way.
    """
    try:
        return bool(torch_npu.npu.aclnn.allow_hf32)
    except AttributeError:
        return True


def schema_arguments(op) -> frozenset[str] | None:
    """Argument names of the installed schema; ``None`` when unreadable.

    Kept in one place so every mHC adapter answers "does this CANN build accept
    the optional operand / switch" the same way.
    """
    packet = getattr(op, "default", None)
    schema = getattr(packet, "_schema", None) if packet is not None else None
    return None if schema is None else frozenset(argument.name for argument in schema.arguments)


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
    """Save the operands the backward of whichever form was called needs.

    torch calls the callback with keyword arguments, spelling the extra
    ``keyword_only_inputs`` only once the schema declares keyword-only
    arguments; both shapes must be accepted and the names must stay exactly as
    torch passes them.  The positional prefix is shared by both forms of the
    operator: ``x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag``.
    ops-transformer appends the optional cross-layer mix as the trailing
    ``pre_mix`` operand; V4 never passes it, V4.1 always does.
    """
    x, phi, alpha, bias = inputs[:4]
    # A caller that leaves the optional trailing operand out keeps the original
    # nine-gradient contract, so the saved layout stays as it was.
    premix = inputs[9] if len(inputs) > 9 else None
    ctx.has_premix = premix is not None
    # The backward owes one gradient per positional input; remember the count so
    # a future schema that appends more optional operands still lines up.
    ctx.num_inputs = len(inputs)
    if premix is None:
        ctx.save_for_backward(x, phi, alpha, bias, *output[3:])
    else:
        ctx.save_for_backward(x, phi, alpha, bias, premix, *output[3:])
    # ``hcEps`` belongs to the stable positional prefix above and never shifts.
    ctx.hc_eps = inputs[6]
    # The trailing outputs are backward-only bookkeeping that the autograd graph
    # never consumes. With the default materialization PyTorch launches a
    # ZerosLike for every unconsumed output on each backward (+5 per iteration),
    # so stop it and let the backward stand None grads in.
    ctx.set_materialize_grads(False)
    if premix is None:
        ctx.mark_non_differentiable(*output[3:])
    else:
        # ``h_pre`` leaves this module as the next sub-layer's ``pre_mix``, so it
        # must stay differentiable; only the four saved intermediates are inert.
        ctx.mark_non_differentiable(*output[4:])


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

    # Materialization is off, so the gradients of outputs the graph does not
    # consume arrive as None; the kernel contract needs zero stand-ins.
    if grad_h_in is None:
        grad_h_in = torch.zeros((*x.shape[:-2], x.shape[-1]), dtype=x.dtype, device=x.device)
    if grad_h_post is None:
        grad_h_post = torch.zeros_like(h_pre)
    if grad_h_res is None:
        grad_h_res = torch.zeros(*x.shape[:2], x.shape[2] * x.shape[2], dtype=h_pre.dtype, device=h_pre.device)

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


def _mhc_pre_sinkhorn_premix_backward(ctx, grad_h_in, grad_h_post, grad_h_res, grad_h_pre, *_):
    """V4.1 form: ``pre_mix`` drives the collapse, so its gradient rides back.

    ``gradPre`` -- the gradient the next sub-layer sends through the exported
    ``pre`` -- is zero-filled when that output went unused downstream, and the
    v2 formula returns ``gradPremix`` for the operand itself.
    """
    (
        x,
        phi,
        alpha,
        bias,
        premix,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
    ) = ctx.saved_tensors
    # Materialization is off, so unconsumed outputs arrive as None; the kernel
    # contract still needs tensors, so stand zeros in for them.
    if grad_h_in is None:
        grad_h_in = torch.zeros((*x.shape[:-2], x.shape[-1]), dtype=x.dtype, device=x.device)
    if grad_h_post is None:
        grad_h_post = torch.zeros_like(h_pre)
    if grad_h_res is None:
        grad_h_res = torch.zeros(*x.shape[:2], x.shape[2] * x.shape[2], dtype=h_pre.dtype, device=h_pre.device)
    grad_pre = grad_h_pre if grad_h_pre is not None else torch.zeros_like(premix)

    grad_x, grad_phi, grad_alpha, grad_bias, grad_premix = torch.ops.cann_ops_transformer.mhc_pre_sinkhorn_backward_v2(
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
        pre_mix=premix,
        grad_pre=grad_pre,
        hc_eps=ctx.hc_eps,
        enable_hf32=allow_hf32(),
    )
    grad_premix_ret = grad_premix if grad_premix.numel() > 0 else None
    return grad_x, grad_phi, grad_alpha, grad_bias, None, None, None, None, None, grad_premix_ret


def _mhc_pre_sinkhorn_autograd_backward(ctx, grad_h_in, grad_h_post, grad_h_res, grad_h_pre=None, *_):
    """One registered backward for both forms of ``mhc_pre_sinkhorn``.

    V4.1 passes ``pre_mix`` and needs the v2 formula, which is the only one
    returning ``gradPremix``; V4 keeps its original v1 formula.  The schema
    shape decides how many gradient slots the answer owes.
    """
    if ctx.has_premix:
        grads = list(_mhc_pre_sinkhorn_premix_backward(ctx, grad_h_in, grad_h_post, grad_h_res, grad_h_pre))
    else:
        # The schema may carry the optional operand even when this call left it
        # out, so pad the nine-slot answer up to the input count.
        grads = list(_mhc_pre_sinkhorn_backward(ctx, grad_h_in, grad_h_post, grad_h_res))
    grads.extend([None] * (ctx.num_inputs - len(grads)))
    return tuple(grads)


torch.library.register_autograd(
    "cann_ops_transformer::mhc_pre_sinkhorn",
    _mhc_pre_sinkhorn_autograd_backward,
    setup_context=_mhc_pre_sinkhorn_setup_context,
)
