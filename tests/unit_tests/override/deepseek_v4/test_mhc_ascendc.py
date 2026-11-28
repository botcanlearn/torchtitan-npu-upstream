# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the fused HcPre override with a CPU stand-in for the CANN kernel."""

from itertools import count
from types import SimpleNamespace

import pytest
import torch
from torchtitan.config import apply_overrides
from torchtitan.models.deepseek_v4.mhc import HcPre

from torchtitan_npu.ops.ascendc import mhc as mhc_ops
from torchtitan_npu.override.deepseek_v4.mhc import ascendc
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

_POSITIONAL_ARGS = (
    "Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
    "int hcMult, int numIters, float hcEps, float normEps, bool outFlag"
)
_OUTPUTS = "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
_NAMESPACES = count()


@pytest.fixture(params=["legacy", "keyword-only"])
def cpu_mhc(request, monkeypatch):
    supports_hf32 = request.param == "keyword-only"
    signature = _POSITIONAL_ARGS
    if supports_hf32:
        signature += ", Tensor? pre_mix=None, *, bool enable_hf32=True"

    forward_calls, autograd_calls, backward_calls = [], [], []
    aclnn = SimpleNamespace(allow_hf32=True)

    def forward(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag, **kwargs):
        forward_calls.append((x.shape, hc_mult, num_iters, hc_eps, norm_eps, out_flag, kwargs))
        batch, tokens = x.shape[:2]
        post = torch.ones(batch, tokens, hc_mult)
        residual = torch.arange(hc_mult**2, dtype=torch.float32).expand(batch, tokens, -1).clone()
        saved = tuple(torch.full_like(post, float(index)) for index in range(5))
        return x[..., 0, :].float().clone(), post, residual, *saved

    def backward(grad_in, grad_post, grad_res, x, phi, alpha, bias, *saved_and_eps):
        backward_calls.append((grad_in, grad_post, grad_res, *saved_and_eps))
        return torch.ones_like(x), torch.full_like(phi, 2.0), torch.full_like(alpha, 3.0), torch.full_like(bias, 4.0)

    def autograd_backward(ctx, *gradients):
        autograd_calls.append(tuple(gradient is not None for gradient in gradients))
        return mhc_ops._mhc_pre_sinkhorn_backward(ctx, *gradients)

    probe = ascendc._mhc_pre_sinkhorn_accepts_hf32
    probe.cache_clear()
    namespace = f"torchtitan_npu_test_mhc_{next(_NAMESPACES)}"
    lib = torch.library.Library(namespace, "FRAGMENT")
    try:
        lib.define(f"mhc_pre_sinkhorn({signature}) -> {_OUTPUTS}")
        lib.impl("mhc_pre_sinkhorn", forward, "CPU")
        op = getattr(torch.ops, namespace).mhc_pre_sinkhorn
        torch.library.register_autograd(
            op.default,
            autograd_backward,
            setup_context=mhc_ops._mhc_pre_sinkhorn_setup_context,
            lib=lib,
        )
        with monkeypatch.context() as patch:
            patch.setattr(torch.ops.cann_ops_transformer, "mhc_pre_sinkhorn", op)
            patch.setattr(torch.ops.cann_ops_transformer, "mhc_pre_sinkhorn_backward", backward, raising=False)
            patch.setattr(ascendc.torch_npu.npu, "aclnn", aclnn)
            yield SimpleNamespace(
                op=op,
                supports_hf32=supports_hf32,
                aclnn=aclnn,
                forward_calls=forward_calls,
                autograd_calls=autograd_calls,
                backward_calls=backward_calls,
            )
    finally:
        probe.cache_clear()


@pytest.fixture
def mhc_pre_model(cpu_mhc):
    config = deepseek_v4_debugmodel()
    config.override.imports = ["torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre"]
    apply_overrides(config.override, config)
    pre = next(node for _, node, _, _ in config.model.traverse(HcPre.Config))
    assert isinstance(pre, ascendc.AscHcPre.Config)
    return pre, pre.build()


def test_mhc_pre_override_preserves_tnd_and_follows_hf32(cpu_mhc, mhc_pre_model):
    pre, model = mhc_pre_model
    x = torch.arange(3 * 4 * pre.dim, dtype=torch.float32).reshape(3, 4, pre.dim).bfloat16()

    for allow_hf32 in (False, True):
        cpu_mhc.aclnn.allow_hf32 = allow_hf32
        h_in, h_post, h_res = model(x)

        shape, hc_mult, iters, hc_eps, norm_eps, save_aux, kwargs = cpu_mhc.forward_calls[-1]
        assert (shape, hc_mult, iters, hc_eps, norm_eps, save_aux) == (
            (1, 3, 4, pre.dim), 4, pre.sinkhorn_iters, pre.eps, pre.norm_eps, True,
        )
        if cpu_mhc.supports_hf32:
            assert kwargs.get("enable_hf32", True) is allow_hf32
        else:
            assert not kwargs
        torch.testing.assert_close(h_in, x[:, 0, :])
        torch.testing.assert_close(h_post, torch.ones(3, 4))
        torch.testing.assert_close(h_res, torch.arange(16, dtype=torch.float32).reshape(4, 4).expand(3, -1, -1))
    assert len(cpu_mhc.forward_calls) == 2


@pytest.mark.parametrize("used_output", [0, 2], ids=["h-in", "h-res"])
def test_mhc_pre_backward_supplies_only_missing_primary_gradients(cpu_mhc, used_output):
    x = torch.ones(1, 3, 4, 8, requires_grad=True)
    phi = torch.ones(24, 32, requires_grad=True)
    alpha = torch.ones(3, requires_grad=True)
    bias = torch.ones(24, requires_grad=True)
    kwargs = {"enable_hf32": False} if cpu_mhc.supports_hf32 else {}

    outputs = cpu_mhc.op(x, phi, alpha, bias, 4, 3, 1e-6, 2e-6, True, **kwargs)
    outputs[used_output].sum().backward()

    assert all(not output.requires_grad for output in outputs[3:])
    assert cpu_mhc.autograd_calls == [tuple(index == used_output for index in range(8))]
    (call,) = cpu_mhc.backward_calls
    for index, (gradient, shape) in enumerate(zip(call[:3], [(1, 3, 8), (1, 3, 4), (1, 3, 16)], strict=True)):
        torch.testing.assert_close(gradient, torch.full(shape, float(index == used_output)))
    for index, saved in enumerate(call[3:-1]):
        torch.testing.assert_close(saved, torch.full((1, 3, 4), float(index)))
    assert len(call[3:-1]) == 5
    assert call[-1] == pytest.approx(1e-6)
    for index, parameter in enumerate((x, phi, alpha, bias), start=1):
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, float(index)))


def test_mhc_pre_bf16_module_backward_uses_kernel_output_dtype(cpu_mhc, mhc_pre_model):
    pre, model = mhc_pre_model
    x = torch.ones(3, pre.hc_mult, pre.dim, dtype=torch.bfloat16, requires_grad=True)

    h_in, _, h_res = model(x)
    h_res.sum().backward()

    assert h_in.dtype == torch.bfloat16
    assert cpu_mhc.autograd_calls == [(False, False, True, False, False, False, False, False)]
    (call,) = cpu_mhc.backward_calls
    grad_in, grad_post, grad_res = call[:3]
    # The kernel returns FP32 h_in before the wrapper casts it back to BF16.
    torch.testing.assert_close(grad_in, torch.zeros((1, 3, pre.dim), dtype=torch.float32))
    torch.testing.assert_close(grad_post, torch.zeros((1, 3, pre.hc_mult), dtype=torch.float32))
    torch.testing.assert_close(grad_res, torch.ones((1, 3, pre.hc_mult**2), dtype=torch.float32))
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    torch.testing.assert_close(model.hc_fn.grad, torch.full_like(model.hc_fn, 2.0))
