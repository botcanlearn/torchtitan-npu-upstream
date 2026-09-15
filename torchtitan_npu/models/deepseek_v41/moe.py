# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 Mixture-of-Experts: the VL router and the golden expert arithmetic.

The router computes FP32 gate scores with the ``sqrtsoftplus`` score
function, applies the image bias (``bias_vl``) on image-masked tokens,
selects the top-k experts in sorted order (the frozen reference digests
depend on the index order), and applies route norm/scale.

All weights use standard dispatch/combine with score-absorbing experts.

The components inherit the public (patched) torchtitan MoE classes; all V4.1
arithmetic lives here.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    RoutedExperts,
    TokenChoiceTopKRouter,
)


def golden_expert(x, w1, w2, w3, route_weights=None, limit=0.0):
    """The reference expert computation: FP32 SwiGLU with the clamp."""
    dtype = x.dtype
    gate = F.linear(x, w1).float()
    up = F.linear(x, w3).float()
    if limit > 0:
        up = up.clamp(min=-limit, max=limit)
        gate = gate.clamp(max=limit)
    hidden = F.silu(gate) * up
    if route_weights is not None:
        hidden = route_weights * hidden
    return F.linear(hidden.to(dtype), w2)


class V41Router(TokenChoiceTopKRouter):
    """TokenChoiceTopKRouter with the V4.1 vision-language bias.

    ``bias_vl`` is a discrete routing parameter: on image-masked tokens it
    replaces the load-balancing bias in the expert choice (it is not
    updated by the balancing hook).  Top-k is sorted (reference order).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        vision_enabled: bool = False
        score_func: str = "sqrtsoftplus"  # pyrefly: ignore [bad-override]

    def __init__(self, config: Config):
        super().__init__(config)
        self.bias_vl = (
            torch.nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32)) if config.vision_enabled else None
        )

    def reset_parameters(self):
        bias_vl = getattr(self, "bias_vl", None)
        if bias_vl is not None:
            torch.nn.init.zeros_(bias_vl)

    def forward(self, x_BLD, expert_bias_E=None, *, input_ids=None, image_mask=None):
        # Compute gate in float32: the reference computes router scores from
        # FP32 activations and weights even when the storage dtype is BF16.
        gate_weight = getattr(self.gate, "weight", None)
        if gate_weight is not None and not isinstance(gate_weight, DTensor):
            scores = F.linear(x_BLD.float(), gate_weight.float())
        else:
            with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
                scores = self.gate(x_BLD)
            # Some eager backends do not implement float32 autocast; promote
            # the gate result explicitly before score transforms and top-k.
            scores = scores.float()
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            # Use the baseline's ``F.softplus`` expression directly. Equivalent
            # formulas can round differently near top-k decision boundaries.
            scores = F.softplus(scores).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        choice_bias = scores.new_zeros(self.num_experts) if expert_bias_E is None else expert_bias_E
        if image_mask is not None and self.bias_vl is not None:
            choice_bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, choice_bias)
        scores_for_choice = scores + choice_bias
        # Apply node-limited routing if configured (upstream behavior).
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        # Sorted top-k: the index order feeds the fp32 route_norm sum and is
        # visible in the frozen reference digests.
        selected_experts_indices = scores_for_choice.topk(self.top_k, dim=-1)[1]

        top_scores = scores.gather(dim=-1, index=selected_experts_indices)

        if self._debug_force_load_balance:
            selected_experts_indices, top_scores = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        return top_scores, selected_experts_indices, scores


class V41RoutedExperts(RoutedExperts):
    """Score-absorbing routed experts with the golden dtype contract.

    The golden experts return FP32 outputs; the combine residual is cast
    to match so the dispatcher's scatter keeps FP32 precision through the
    combine (the single BF16 cast happens once at the MoE output).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = x_BLD.shape
        K = topk_scores_BLK.size(-1)
        T = B * L
        x_TD = x_BLD.view(T, D)

        topk_scores_TK = topk_scores_BLK.view(T, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(T, K)
        dispatcher = self.token_dispatcher
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        routed_scores_R = getattr(metadata, "routed_scores_R", None)

        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(
                routed_input_RD,
                num_global_tokens_per_local_expert_e,
                routed_scores_R=routed_scores_R,
            )

        out_TD = dispatcher.combine(routed_output_RD, metadata, x_TD.to(routed_output_RD.dtype))
        return out_TD.view(B, -1, D)


class V41GroupedExperts(GroupedExperts):
    """The golden per-expert computation with the score absorbed pre-W2.

    Empty experts stay in the autograd graph (every shard gets a gradient).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        *,
        routed_scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = [
            weight.to_local() if isinstance(weight, DTensor) else weight
            for weight in (self.w1_EFD, self.w2_EDF, self.w3_EFD)
        ]
        counts = num_tokens_per_expert_E.tolist()
        parts = []
        start = 0
        for expert, count in enumerate(counts):
            end = start + count
            # Keep empty expert operations in autograd so every shard gets a gradient.
            score = None if routed_scores_R is None else routed_scores_R[start:end, None]
            parts.append(
                golden_expert(
                    x_RD[start:end],
                    *(weight[expert] for weight in weights),
                    route_weights=score,  # pyrefly: ignore [bad-keyword-argument]
                    limit=self.swiglu_limit,  # pyrefly: ignore [bad-keyword-argument]
                ).float()
            )
            start = end
        return torch.cat(parts, dim=0)


class V41FeedForward(FeedForward):
    """The golden shared-expert computation."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        pass

    def forward(self, x):
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:  # pyrefly: ignore [unsupported-operation]
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)  # pyrefly: ignore [unsupported-operation]
        return self.w2((F.silu(gate) * up).to(dtype)).float()


class V41MoE(MoE):
    """V4.1 MoE: the VL router plus the golden expert execution.

    ``Config`` is redefined so ``build()`` instantiates this class rather
    than the upstream ``MoE`` (the inherited alias would construct the
    base).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def forward(
        self,
        x_BLD: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _B, L, _D = x_BLD.shape
        sp_size = getattr(self.routed_experts.token_dispatcher, "sp_size", 1)
        if not isinstance(x_BLD, DTensor) and getattr(self, "seq_dim_tp_sharded", False):
            seq_pad = 0
            seq_dim_pad_tokens = 0
        else:
            seq_pad = sp_size - L if sp_size > L else 0
            if seq_pad:
                x_BLD = F.pad(x_BLD, (0, 0, 0, seq_pad))
                L = L + seq_pad
            seq_dim_pad_tokens = (-L) % sp_size

        (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        ) = self.router(
            x_BLD,
            getattr(self, "expert_bias_E", None),
            input_ids=input_ids,
            image_mask=image_mask,
        )

        routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_BLK,
            True,
        )
        num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))

        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_BLD = self.routed_experts(
            x_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            num_local_tokens_per_expert_E,
        )

        shared_out_BLD = self.shared_experts(x_BLD) if self.shared_experts is not None else None

        if shared_out_BLD is not None:
            out_BLD = out_BLD + shared_out_BLD

        if seq_dim_pad_tokens:
            out_BLD = out_BLD[:, :L, :]

        if seq_pad:
            out_BLD = out_BLD[:, : L - seq_pad, :]

        return out_BLD.to(x_BLD.dtype)
