# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 mHC overrides; both preserve the cross-layer pre-mix graph.

``asc_sinkhorn`` fuses the Sinkhorn projection only; ``asc_hc_premix`` fuses the
coefficient derivation, the projection and the collapse through
``cann_ops_transformer.mhc_pre_sinkhorn`` and its optional ``pre_mix`` operand
(``hc_mult=4``, ``sinkhorn_iters=20``, and a last dim the kernel accepts). Both
select the same ``HcPre.Config`` node, so at most one is enabled at a time.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.mhc import HcPost, HcPre
from torchtitan_npu.ops.ascendc.mhc import allow_hf32, schema_arguments


class AscV41HcPre(HcPre):
    """Fuse only Sinkhorn; inherited collapse still uses the caller's pre mix."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def _split_sinkhorn(self, mixes_BLM):
        n = self.hc_mult
        pre, post, comb = mixes_BLM.split([n, n, n * n], dim=-1)
        pre = torch.sigmoid(pre * self.hc_scale[0] + self.hc_base[:n]) + self.hc_eps
        post = 2 * torch.sigmoid(post * self.hc_scale[1] + self.hc_base[n : 2 * n])
        comb = (comb * self.hc_scale[2] + self.hc_base[2 * n :]).unflatten(-1, (n, n)).contiguous()
        comb, _, _ = torch_npu.npu_mhc_sinkhorn(comb, eps=self.hc_eps, num_iters=self.sinkhorn_iters, out_flag=1)
        return pre, post, comb


class AscV41HcPrePremix(HcPre):
    """V4.1 single-pass mHC fused into AscV41HcPrePremix.

    The operator replaces ``asc_sinkhorn``'s job in one call: it derives the
    coefficients, applies the Sinkhorn projection and collapses the streams, so
    this adapter replaces ``forward`` outright instead of only ``_split_sinkhorn``.

    Training drives the same kernel. V4.1 hands the previous sub-layer's ``pre``
    in as ``premix`` and consumes the ``pre`` this layer exports for the next
    one; the kernel's backward takes ``gradPre`` (the gradient arriving at that
    exported ``pre``) and returns ``gradPremix``, so the parameters that only
    shape ``pre`` -- ``hc_scale[0]``, ``hc_base[:hc_mult]`` and
    ``hc_fn[:hc_mult]`` -- stay trainable.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # Reading the schema per call breaks ``torch.compile``: Dynamo cannot
        # iterate the C++ argument list.  Ask once and branch on a plain bool.
        arguments = schema_arguments(torch.ops.cann_ops_transformer.mhc_pre_sinkhorn)
        self._accepts_hf32 = arguments is not None and "enable_hf32" in arguments

    def forward(
        self, x_BLHcD: torch.Tensor, pre_mix_BLHc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The kernel collapses with the cross-layer mix and returns ``hin``
        # already reduced to one stream. The mixing parameters stay fp32 for the
        # kernel even when mixed precision casts the module.
        h_in, h_post, h_res, h_pre, *_ = torch.ops.cann_ops_transformer.mhc_pre_sinkhorn(
            x_BLHcD,
            self.hc_fn.float(),
            self.hc_scale.float(),
            self.hc_base.float(),
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
            self.norm_eps,
            True,  # outFlag: hPre must stay valid for the next sub-layer
            pre_mix_BLHc,  # cross-layer mix; the unified schema keeps it last
            # The shipped schema leaves HF32 to the dispatch, which reads the
            # switch itself; only builds that declare it take it explicitly.
            **({"enable_hf32": allow_hf32()} if self._accepts_hf32 else {}),
        )
        h_res = h_res.reshape(*x_BLHcD.shape[:-2], self.hc_mult, self.hc_mult)
        return h_in.to(x_BLHcD.dtype), h_pre, h_post, h_res


class AscV41HcPost(HcPost):
    @dataclass(kw_only=True, slots=True)
    class Config(HcPost.Config):
        pass

    def forward(self, y_BLD, residual_BLHcD, post_BLHc, comb_BLHcHc):
        if y_BLD.ndim == 2:
            y_BLD = y_BLD.reshape(residual_BLHcD.shape[0], residual_BLHcD.shape[1], -1)
        return torch.ops.cann_ops_transformer.mhc_post(residual_BLHcD, comb_BLHcHc, y_BLD, post_BLHc)


@override(target=HcPre.Config, exact=True, description="V4.1 single-pass mHC with native Sinkhorn")
def asc_sinkhorn(cfg: HcPre.Config) -> AscV41HcPre.Config:
    return derive(cfg, AscV41HcPre.Config)


# The operator is specialised for these two values; anything else
# changes the arithmetic without saying so.
_PREMIX_HC_MULT = 4
_PREMIX_SINKHORN_ITERS = 20


def _require_premix_operator() -> None:
    """Reject ``asc_hc_premix`` before the model is built when it cannot run."""
    ops_ns = getattr(torch.ops, "cann_ops_transformer", None)
    forward = None if ops_ns is None else getattr(ops_ns, "mhc_pre_sinkhorn", None)
    if forward is None:
        raise RuntimeError(
            "mhc.asc_hc_premix needs mhc_pre_sinkhorn from cann_ops_transformer, "
            "which this CANN build does not provide; use mhc.asc_sinkhorn instead"
        )
    # A nine-argument ``mhc_pre_sinkhorn`` predates the cross-layer mix; the
    # probe keeps a mixed build from failing on the first mHC call instead.
    arguments = schema_arguments(forward)
    if arguments is not None and "pre_mix" not in arguments:
        raise RuntimeError(
            "mhc.asc_hc_premix needs the mhc_pre_sinkhorn schema with the optional pre_mix "
            "operand; this CANN build registers the nine-argument form only, "
            "use mhc.asc_sinkhorn instead"
        )
    if getattr(ops_ns, "mhc_pre_sinkhorn_backward_v2", None) is None:
        raise RuntimeError(
            "mhc.asc_hc_premix needs mhc_pre_sinkhorn_backward_v2 to train the premix "
            "parameter, which this CANN build does not provide; use mhc.asc_sinkhorn instead"
        )


@override(
    target=HcPre.Config,
    exact=True,
    description="V4.1 single-pass mHC fused into mhc_pre_sinkhorn (pre_mix path)",
)
def asc_hc_premix(cfg: HcPre.Config) -> AscV41HcPrePremix.Config:
    # Checked here rather than in forward(): the factory sees the model config
    # and runs before build(), so a mismatched recipe fails at startup instead
    # of on the first mHC call.
    if (cfg.hc_mult, cfg.sinkhorn_iters) != (_PREMIX_HC_MULT, _PREMIX_SINKHORN_ITERS):
        raise ValueError(
            f"mhc.asc_hc_premix requires hc_mult={_PREMIX_HC_MULT} and "
            f"sinkhorn_iters={_PREMIX_SINKHORN_ITERS}, got "
            f"(hc_mult={cfg.hc_mult}, sinkhorn_iters={cfg.sinkhorn_iters}); "
            "use mhc.asc_sinkhorn instead"
        )
    _require_premix_operator()
    return derive(cfg, AscV41HcPrePremix.Config)


@override(target=HcPost.Config, exact=True, description="V4.1 AscendC mHC post")
def asc_hc_post(cfg: HcPost.Config) -> AscV41HcPost.Config:
    return derive(cfg, AscV41HcPost.Config)
