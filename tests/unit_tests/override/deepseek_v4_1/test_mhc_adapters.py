# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter contracts for the V4.1 mHC fused modules.

Selection/isolation via the real debug model; arithmetic contracts run on
CPU with the fused kernels replaced by reference implementations.  The
kernel mocks prove the adapter feeds the kernels the documented operands
and keeps the autograd chain connected; NPU kernel execution is validated
on the target hardware (see the verification records).
"""

import pytest
import torch
from torchtitan.config import OverrideConfig
from torchtitan.config.override import apply_overrides

from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre

MHC_POST_OVERRIDE = "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post"
MHC_SINKHORN_OVERRIDE = "torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn"
MHC_PREMIX_OVERRIDE = "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_premix"


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield


class TestMHcAdapters:
    def test_override_selects_and_preserves_config(self):
        cfg = model_registry("deepseek_v4_1_debugmodel").model
        reference = model_registry("deepseek_v4_1_debugmodel").model
        apply_overrides(
            OverrideConfig(imports=[MHC_POST_OVERRIDE, MHC_SINKHORN_OVERRIDE]),
            cfg,
        )
        from torchtitan_npu.override.deepseek_v4_1.mhc import AscV41HcPost, AscV41HcPre

        for layer, ref_layer in zip(cfg.layers[:6], reference.layers[:6], strict=True):
            assert isinstance(layer.hc_post, AscV41HcPost.Config)
            assert isinstance(layer.hc_attn_pre, AscV41HcPre.Config)
            # the fused configs are pure replacements: every declared field
            # keeps the reference value
            for field in ("hc_mult", "dim", "sinkhorn_iters", "hc_eps", "norm_eps"):
                assert getattr(layer.hc_attn_pre, field) == getattr(ref_layer.hc_attn_pre, field)

    def test_hc_post_matches_reference_arithmetic(self, monkeypatch):
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPost.Config().build()
        fused_module = fused.AscV41HcPost.Config().build()
        # Block-level contract: ``y`` is the collapsed single stream [B, T, D];
        # ``residual`` is the multi-stream [B, T, n, D].
        B, T, D, n = 2, 5, 8, 4
        y = torch.randn(B, T, D, dtype=torch.bfloat16)
        residual = torch.randn(B, T, n, D, dtype=torch.bfloat16)
        post = torch.rand(B, T, n, dtype=torch.float32)
        comb = torch.rand(B, T, n, n, dtype=torch.float32)

        def mhc_post(residual, comb, y, post):
            out = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.sum(
                comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
            )
            return out.type_as(y)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_post", mhc_post)

        # Forward and backward both flow through the mocked kernel unchanged:
        # same values, same gradients on every operand.
        ref_inputs = [t.detach().clone().requires_grad_(True) for t in (y, residual, post, comb)]
        fused_inputs = [t.detach().clone().requires_grad_(True) for t in (y, residual, post, comb)]
        expected = reference(*ref_inputs)
        actual = fused_module(*fused_inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.randn_like(expected)
        expected.backward(grad)
        actual.backward(grad)
        for ref_t, fused_t, name in zip(ref_inputs, fused_inputs, ("y", "residual", "post", "comb"), strict=True):
            assert ref_t.grad is not None and fused_t.grad is not None, name
            torch.testing.assert_close(fused_t.grad, ref_t.grad, rtol=1e-5, atol=1e-6)

    def test_hc_post_unfolds_the_batch_folded_stream(self, monkeypatch):
        """Sequence-parallel linears return the collapsed stream as ``[T, D]``;
        the reference broadcasts that silently, the kernel needs ``[B, T, D]``."""
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        fused_module = fused.AscV41HcPost.Config().build()
        B, T, D, n = 1, 5, 8, 4
        y_folded = torch.randn(T, D, dtype=torch.bfloat16)
        residual = torch.randn(B, T, n, D, dtype=torch.bfloat16)
        post = torch.rand(B, T, n, dtype=torch.float32)
        comb = torch.rand(B, T, n, n, dtype=torch.float32)
        expected = HcPost.Config().build()(y_folded.unsqueeze(0), residual, post, comb)

        received = []

        def mhc_post(x, h_res, h_out, h_post):
            received.append(tuple(t.dim() for t in (x, h_res, h_out, h_post)))
            out = h_post.unsqueeze(-1) * h_out.unsqueeze(-2) + torch.sum(
                h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=2
            )
            return out.type_as(h_out)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_post", mhc_post)
        actual = fused_module(y_folded, residual, post, comb)
        assert received[0] == (4, 4, 3, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_split_sinkhorn_matches_reference_and_feeds_kernel(self, monkeypatch):
        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPre.Config(dim=8).build()
        module = fused.AscV41HcPre.Config(dim=8).build()
        with torch.no_grad():
            for parameter in reference.parameters():
                torch.nn.init.trunc_normal_(parameter, std=0.02)
        module.load_state_dict(reference.state_dict())
        n = module.hc_mult

        calls = []

        def sinkhorn(comb, *, eps, num_iters, out_flag):
            # Differentiable CPU stand-in mirroring the reference's balancing:
            # proving the adapter connects the graph, not the native kernel.
            calls.append((eps, num_iters, out_flag))
            row_max = comb.max(dim=-1, keepdim=True).values
            out = torch.exp(comb - row_max)
            out = out / out.sum(dim=-1, keepdim=True) + eps
            out = out / (out.sum(dim=-2, keepdim=True) + eps)
            for _ in range(num_iters - 1):
                out = out / (out.sum(dim=-1, keepdim=True) + eps)
                out = out / (out.sum(dim=-2, keepdim=True) + eps)
            return out, None, None

        monkeypatch.setattr(fused.torch_npu, "npu_mhc_sinkhorn", sinkhorn)

        B, T = 2, 3
        mixes = torch.randn(B, T, 2 * n + n * n, requires_grad=True)
        ref_mixes = mixes.detach().clone().requires_grad_(True)
        ref_pre, ref_post, ref_comb = reference._split_sinkhorn(ref_mixes)
        pre, post, comb = module._split_sinkhorn(mixes)

        # pre/post keep the reference arithmetic exactly; comb matches the
        # stand-in's balancing (same math as the reference), and the kernel
        # received the documented iteration knobs.
        torch.testing.assert_close(pre, ref_pre, rtol=0, atol=0)
        torch.testing.assert_close(post, ref_post, rtol=0, atol=0)
        torch.testing.assert_close(comb, ref_comb, rtol=1e-6, atol=1e-7)
        assert calls[0] == (module.hc_eps, module.sinkhorn_iters, 1)

        # the gradient connection: identical backward through both paths
        (pre.sum() + post.sum() + comb.sum()).backward()
        (ref_pre.sum() + ref_post.sum() + ref_comb.sum()).backward()
        torch.testing.assert_close(mixes.grad, ref_mixes.grad, rtol=1e-5, atol=1e-7)

    def test_premix_override_selects_premix_adapter(self):
        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused
        from torchtitan_npu.override.deepseek_v4_1.mhc import AscV41HcPrePremix

        cfg = model_registry("deepseek_v4_1_debugmodel").model
        try:
            fused._require_premix_operator()
        except RuntimeError:
            # A CANN build without the unified operator must reject the entry at
            # build time instead of failing on the first mHC call.
            with pytest.raises(RuntimeError, match="use mhc.asc_sinkhorn"):
                apply_overrides(OverrideConfig(imports=[MHC_PREMIX_OVERRIDE]), cfg)
            return

        apply_overrides(OverrideConfig(imports=[MHC_PREMIX_OVERRIDE]), cfg)
        for layer in cfg.layers[:6]:
            assert isinstance(layer.hc_attn_pre, AscV41HcPrePremix.Config)
            assert isinstance(layer.hc_ffn_pre, AscV41HcPrePremix.Config)

    def test_premix_adapter_uses_the_kernel_in_inference(self, monkeypatch):
        """Inference has no gradient to preserve, so the kernel collapses."""
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPre.Config(dim=8).build()
        with torch.no_grad():
            for parameter in reference.parameters():
                torch.nn.init.trunc_normal_(parameter, std=0.02)
        module = fused.AscV41HcPrePremix.Config(dim=8).build()
        module.load_state_dict(reference.state_dict())

        batch, seq, dim, n = 2, 3, 8, module.hc_mult
        x = torch.randn(batch, seq, n, dim, dtype=torch.bfloat16)
        pre_mix = torch.full((batch, seq, n), 0.25, dtype=torch.float32)
        seen = {}

        def premix_op(
            x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag, premix, **hf32_kwargs
        ):
            seen["hf32_kwargs"] = hf32_kwargs
            seen["premix"] = premix
            seen["out_flag"] = out_flag
            pre = torch.full((batch, seq, n), 0.25, dtype=torch.float32)
            post = torch.ones(batch, seq, n, dtype=torch.float32)
            comb = torch.full((batch, seq, n, n), 1.0 / n, dtype=torch.float32)
            hin = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)
            empty = torch.empty(0)
            return (hin, post, comb.reshape(batch, seq, n * n), pre, empty, empty, empty, empty)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_pre_sinkhorn", premix_op)

        with torch.no_grad():
            y, pre, post, comb = module(x, pre_mix)
        assert seen["premix"] is pre_mix
        assert seen["out_flag"] is True
        torch.testing.assert_close(y, module.collapse(x, pre_mix), rtol=0, atol=0)
        assert comb.shape == (batch, seq, n, n)

    def test_premix_adapter_drives_the_kernel_in_training(self, monkeypatch):
        """The kernel runs under autograd, keeping ``hPre`` valid for the next layer."""
        import torch as torch_mod

        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        reference = HcPre.Config(dim=8).build()
        with torch.no_grad():
            for parameter in reference.parameters():
                torch.nn.init.trunc_normal_(parameter, std=0.02)
        module = fused.AscV41HcPrePremix.Config(dim=8).build()
        module.load_state_dict(reference.state_dict())

        batch, seq, dim, n = 2, 3, 8, module.hc_mult
        x = torch.randn(batch, seq, n, dim, dtype=torch.bfloat16)
        pre_mix = torch.full((batch, seq, n), 0.25, dtype=torch.float32)
        seen = {}

        def premix_op(x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag, premix):
            seen["premix"] = premix
            seen["out_flag"] = out_flag
            pre = torch.full((batch, seq, n), 0.25, dtype=torch.float32)
            post = torch.ones(batch, seq, n, dtype=torch.float32)
            comb = torch.full((batch, seq, n, n), 1.0 / n, dtype=torch.float32)
            hin = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)
            empty = torch.empty(0)
            return (hin, post, comb.reshape(batch, seq, n * n), pre, empty, empty, empty, empty)

        monkeypatch.setattr(torch_mod.ops.cann_ops_transformer, "mhc_pre_sinkhorn", premix_op)

        y, pre, post, comb = module(x, pre_mix)

        assert seen["premix"] is pre_mix
        assert seen["out_flag"] is True
        torch.testing.assert_close(y, module.collapse(x, pre_mix), rtol=0, atol=0)
        assert comb.shape == (batch, seq, n, n)

    # The forward tests above replace the op with a plain Python callable, so
    # ``register_autograd`` never runs and the backward has no coverage there.
    # These cases call it directly with a fake ctx and a recording kernel: they
    # pin the operand order, the gradient-slot alignment, and every "missing
    # gradient" branch.

    @staticmethod
    def _premix_backward_fixture(*, with_premix: bool, slot: bool | None = None):
        from types import SimpleNamespace

        batch, seq, n, dim = 2, 3, 4, 8
        mix = (2 + n) * n
        premix = torch.randn(batch, seq, n) if with_premix else None
        if slot is None:
            slot = with_premix
        saved = (
            torch.randn(batch, seq, n, dim),  # x
            torch.randn(mix, n * dim),  # phi
            torch.randn(3),  # alpha
            torch.randn(mix),  # bias
            # V4 keeps the original nine-tensor layout; V4.1 stores ``premix``.
            *((premix,) if with_premix else ()),
            torch.randn(batch, seq, n),  # h_pre
            torch.randn(batch, seq, n),  # hc_before_norm
            torch.randn(batch, seq),  # inv_rms
            torch.randn(batch, seq),  # sum_out
            torch.randn(batch, seq),  # norm_out
        )
        ctx = SimpleNamespace(
            saved_tensors=saved,
            hc_eps=1e-6,
            has_premix=with_premix,
            num_inputs=10 if slot else 9,
        )
        return ctx, saved

    @staticmethod
    def _install_recording_premix_backward(monkeypatch, *, empty_grad_premix: bool = False):
        """Stub ``mhc_pre_sinkhorn_backward_v2`` (the V4.1 formula)."""
        import torch as torch_mod

        recorded = {}

        def premix_backward(*args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            premix_arg = kwargs.get("pre_mix")
            if premix_arg is not None and not empty_grad_premix:
                grad_premix = torch.full_like(premix_arg, 0.5)
            else:
                grad_premix = torch.empty(0)
            return (
                torch.zeros_like(args[3]),  # grad_x
                torch.zeros_like(args[4]),  # grad_phi
                torch.zeros_like(args[5]),  # grad_alpha
                torch.zeros_like(args[6]),  # grad_bias
                grad_premix,
            )

        # ``raising=False``: the op only exists once the CANN package ships it and
        # this stub replaces it either way.
        monkeypatch.setattr(
            torch_mod.ops.cann_ops_transformer,
            "mhc_pre_sinkhorn_backward_v2",
            premix_backward,
            raising=False,
        )
        return recorded

    @staticmethod
    def _install_recording_v4_backward(monkeypatch):
        """Stub ``mhc_pre_sinkhorn_backward`` (the untouched V4 formula)."""
        import torch as torch_mod

        recorded = {}

        def v4_backward(*args):
            recorded["args"] = args
            return (
                torch.zeros_like(args[3]),  # grad_x
                torch.zeros_like(args[4]),  # grad_phi
                torch.zeros_like(args[5]),  # grad_alpha
                torch.zeros_like(args[6]),  # grad_bias
            )

        monkeypatch.setattr(
            torch_mod.ops.cann_ops_transformer,
            "mhc_pre_sinkhorn_backward",
            v4_backward,
            raising=False,
        )
        return recorded

    def test_premix_backward_aligns_operands_and_gradient_slots(self, monkeypatch):
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, saved = self._premix_backward_fixture(with_premix=True)
        recorded = self._install_recording_premix_backward(monkeypatch)
        grad_h_pre = torch.randn(2, 3, 4)
        # The backward must follow the same run-wide switch the forward reads.
        monkeypatch.setattr(mhc_ops, "allow_hf32", lambda: False)

        grads = mhc_ops._mhc_pre_sinkhorn_autograd_backward(
            ctx, torch.randn(2, 3, 8), torch.randn(2, 3, 4), torch.randn(2, 3, 16), grad_h_pre, None
        )

        args = recorded["args"]
        kwargs = recorded["kwargs"]
        # 12 positional tensors; the premix pair travels keyword-only.
        assert len(args) == 12
        assert kwargs["pre_mix"] is saved[4]
        assert kwargs["grad_pre"] is grad_h_pre
        assert kwargs["hc_eps"] == ctx.hc_eps
        assert kwargs["enable_hf32"] is False
        # 10 autograd slots: 4 parameter grads, 5 attribute slots, then pre_mix.
        assert len(grads) == 10
        assert grads[4:9] == (None,) * 5
        torch.testing.assert_close(grads[9], torch.full_like(saved[4], 0.5))

    def test_setup_context_keeps_the_premix_operand(self):
        """The real setup path (not a hand-built ctx) stores the operand."""
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        class _Context:
            def save_for_backward(self, *tensors):
                self.saved = tensors

            def set_materialize_grads(self, enabled):
                self.materialize_grads = enabled

            def mark_non_differentiable(self, *tensors):
                self.non_differentiable = tensors

        ctx = _Context()
        x = torch.randn(2, 3, 4, 8)
        phi = torch.randn(24, 32)
        alpha = torch.randn(3)
        bias = torch.randn(24)
        premix = torch.randn(2, 3, 4)
        outputs = [torch.zeros(1) for _ in range(8)]

        mhc_ops._mhc_pre_sinkhorn_setup_context(
            ctx=ctx,
            inputs=[x, phi, alpha, bias, 4, 20, 1e-6, 2e-6, True, premix],
            output=outputs,
        )

        # ``x, phi, alpha, bias, premix`` then the five saved intermediates.
        assert len(ctx.saved) == 10
        assert ctx.saved[4] is premix
        assert ctx.has_premix is True
        assert ctx.num_inputs == 10
        assert ctx.hc_eps == 1e-6
        assert ctx.materialize_grads is False
        assert ctx.non_differentiable == tuple(outputs[4:])

    def test_premix_adapter_probes_hf32_once_at_construction(self, monkeypatch):
        """``torch.compile`` cannot read the schema per call, so probe at build time."""
        from torchtitan_npu.override.deepseek_v4_1 import mhc as fused

        monkeypatch.setattr(fused, "schema_arguments", lambda op: frozenset({"x", "pre_mix", "enable_hf32"}))
        assert fused.AscV41HcPrePremix.Config(dim=8).build()._accepts_hf32 is True

        # The shipped schema leaves HF32 to the dispatch: stay silent.
        monkeypatch.setattr(fused, "schema_arguments", lambda op: frozenset({"x", "pre_mix"}))
        assert fused.AscV41HcPrePremix.Config(dim=8).build()._accepts_hf32 is False

        # An unreadable schema must not turn into an unexpected keyword argument.
        monkeypatch.setattr(fused, "schema_arguments", lambda op: None)
        assert fused.AscV41HcPrePremix.Config(dim=8).build()._accepts_hf32 is False

    def test_premix_backward_zero_fills_missing_output_gradients(self, monkeypatch):
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, saved = self._premix_backward_fixture(with_premix=True)
        recorded = self._install_recording_premix_backward(monkeypatch)

        mhc_ops._mhc_pre_sinkhorn_autograd_backward(
            ctx, None, torch.randn(2, 3, 4), torch.randn(2, 3, 16), None, None
        )

        args = recorded["args"]
        x = saved[0]
        # hin took no upstream gradient: it is zero-filled, not skipped.
        assert args[0].shape == (x.shape[0], x.shape[1], x.shape[-1])
        assert args[0].dtype == x.dtype
        assert torch.count_nonzero(args[0]).item() == 0
        # pre was unused downstream: gradPre is zero-filled so the kernel still
        # receives a tensor and the parameters keep their hIn-path gradient.
        torch.testing.assert_close(recorded["kwargs"]["grad_pre"], torch.zeros_like(saved[4]))

    def test_premix_backward_drops_empty_kernel_gradient(self, monkeypatch):
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, _ = self._premix_backward_fixture(with_premix=True)
        self._install_recording_premix_backward(monkeypatch, empty_grad_premix=True)

        grads = mhc_ops._mhc_pre_sinkhorn_autograd_backward(
            ctx, torch.randn(2, 3, 8), torch.randn(2, 3, 4), torch.randn(2, 3, 16), torch.randn(2, 3, 4), None
        )

        assert grads[9] is None

    def test_absent_premix_keeps_the_v4_formula(self, monkeypatch):
        """A ten-slot schema without the operand still answers with v1."""
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, _ = self._premix_backward_fixture(with_premix=False, slot=True)
        recorded = self._install_recording_v4_backward(monkeypatch)

        grads = mhc_ops._mhc_pre_sinkhorn_autograd_backward(
            ctx, torch.randn(2, 3, 8), torch.randn(2, 3, 4), torch.randn(2, 3, 16), None, None
        )

        # v1 takes hc_eps positionally, right after norm_out.
        assert len(recorded["args"]) == 13
        assert recorded["args"][12] == ctx.hc_eps
        # The schema still owns the trailing slot, so it is padded with ``None``.
        assert len(grads) == 10
        assert grads[9] is None

    def test_backward_without_premix_keeps_the_v4_contract(self, monkeypatch):
        from torchtitan_npu.ops.ascendc import mhc as mhc_ops

        ctx, _ = self._premix_backward_fixture(with_premix=False, slot=False)
        recorded = self._install_recording_v4_backward(monkeypatch)

        grads = mhc_ops._mhc_pre_sinkhorn_autograd_backward(
            ctx, torch.randn(2, 3, 8), torch.randn(2, 3, 4), torch.randn(2, 3, 16), None, None
        )

        # Nine slots: the V4 schema has no pre_mix operand.
        assert len(grads) == 9
        assert grads[4:] == (None,) * 5
        assert recorded["args"][12] == ctx.hc_eps
