# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

from torch import nn, Tensor

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.npu_registry import register_model_converter
from torchtitan_npu.models.deepseek_v4.model import HcHead, HcPost, HcPre
from torchtitan_npu.ops.triton import MHCPostTriton, MHCPreOnlyTriton, MHCPreTriton


logger = logging.getLogger(__name__)


class NpuHcPre(HcPre):
    def __init__(self, parent: HcPre):
        # Shallow copy of parent's __dict__ is intentional here:
        # - HcPre attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on HcPre.__init__ parameters (hc_mult, hc_sinkhorn_iters, etc.)
        # - Parent instance already has all attributes properly initialized
        # Note: If HcPre had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ):
        r"""HcPre forward using Triton implementation.


        This function executes the "Pre-Mapping" stage of the mHC architecture. It first flattens
        the input from 4D to 3D, then applies RMSNorm normalization, computes manifold-constrained
        connection weights (`h_pre`, `h_post`, `h_res`) via linear projection and the Sinkhorn-Knopp
        algorithm, and finally aggregates the input using `h_pre` to generate the main branch output.


        Args:
            self: Module instance containing hc_mult, hc_sinkhorn_iters, hc_eps attributes
            x (torch.Tensor):
                Input tensor of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.
            hc_fn (torch.Tensor):
                Projection weight matrix of shape `[n * n + 2 * n, n * D]`.
                Used to map input to the hyper-connection space.
            hc_scale (torch.Tensor):
                Branch Alpha parameters of shape `[3]`.
            hc_base (torch.Tensor):
                Branch Beta parameters of shape `[2 * n + n * n]`.


        Returns:
            y (torch.Tensor):
                Main branch output of shape `[B, S, D]`.
            h_post (torch.Tensor):
                Post-processing weight matrix of shape `[B, S, n]`.
            h_res (torch.Tensor):
                Residual weight matrix of shape `[B, S, n, n]`.
        """
        x = x.flatten(2)

        y, h_post, h_res = MHCPreTriton.apply(
            x,  # x
            hc_fn,  # weight
            hc_scale,  # branch_alpha
            hc_base,  # branch_beta
            None,  # norm_gamma
            False,  # mhc_use_gamma
            self.hc_mult,  # num_stream
            self.hc_sinkhorn_iters,  # sinkhorn_iters
            self.hc_eps,  # eps
        )
        return y, h_post, h_res


class NpuHcPost(HcPost):
    def __init__(self, parent: HcPost):
        # Shallow copy of parent's __dict__ is intentional here:
        # - HcPost attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on HcPost.__init__ parameters
        # - Parent instance already has all attributes properly initialized
        # Note: If HcPost had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        x: Tensor,
        residual: Tensor,
        post: Tensor,
        comb: Tensor,
    ):
        r"""NpuHcPost forward using Triton implementation.


        This function executes the "Post-Mapping" stage of the mHC architecture. It flattens the
        residual from 4D to 3D, then utilizes the weights generated in the pre-stage (`h_post` and `h_res`)
        to perform a manifold-constrained weighted fusion of the current input `x` and the `residual`.


        Args:
            self: Module instance
            x (torch.Tensor):
                Current layer main input of shape `[B, S, D]`.
            residual (torch.Tensor):
                Residual input of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.
            post (torch.Tensor):
                Post-processing weights of shape `[B, S, n]`.
            comb (torch.Tensor):
                Residual weights of shape `[B, S, n, n]`.


        Returns:
            y (torch.Tensor):
                Fused output tensor of shape `[B, S, N, D]`.
        """
        dim_b, dim_s, dim_n, dim_d = residual.shape
        residual = residual.flatten(2)

        y = MHCPostTriton.apply(
            x,  # x
            residual,  # residual
            post,  # h_post
            comb,  # h_res
        )

        y = y.view(dim_b, dim_s, dim_n, dim_d)
        return y


class NpuHcHead(HcHead):
    def __init__(self, parent: HcHead):
        # Shallow copy of parent's __dict__ is intentional here:
        # - HcHead attributes are primarily PyTorch modules and buffers (weights should be shared)
        # - Avoids complex dependency on HcHead.__init__ parameters (norm_eps, hc_eps)
        # - Parent instance already has all attributes properly initialized
        # Note: If HcHead had mutable non-module attributes requiring independent state,
        # we would need explicit attribute copying instead
        self.__dict__.update(parent.__dict__)

    def forward(
        self,
        x: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ):
        r"""Lightweight MHC Pre-Aggregation Function (Head forward).


        Similar to `hc_pre`, but this function does not return the intermediate Sinkhorn states
        (`h_post`, `h_res`), returning only the weighted aggregated output. The input is flattened
        from 4D to 3D before processing.


        Args:
            self: Module instance containing hc_mult, hc_eps attributes
            x (torch.Tensor):
                Input tensor of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.
            hc_fn (torch.Tensor):
                Projection weight matrix of shape `[n, n * D]`.
            hc_scale (torch.Tensor):
                Branch Alpha parameters of shape `[1]`.
            hc_base (torch.Tensor):
                Branch Beta parameters of shape `[n]`.


        Returns:
            y (torch.Tensor):
                Weighted aggregated output of shape `[B, S, D]`.
        """
        x = x.flatten(2)

        y = MHCPreOnlyTriton.apply(
            x,  # x
            hc_fn,  # weight
            hc_scale,  # branch_alpha
            hc_base,  # branch_beta
            None,  # norm_gamma
            False,  # mhc_use_gamma
            self.hc_eps,  # eps
        )
        return y


class MHCPreConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, HcPre):
                replace_module_with_name(model, name, NpuHcPre(module))
                logger.info("[MHCPreConverter] [HcPre forward] Applied.")


@register_model_converter("npu_mhc_pre")
class MHCPrePostModelConfig(ModelCustomConfig):
    model_converter = MHCPreConverter


class MHCPostConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, HcPost):
                replace_module_with_name(model, name, NpuHcPost(module))
                logger.info("[MHCPostConverter] [HcPost forward] Applied.")

            if isinstance(module, HcHead):
                replace_module_with_name(model, name, NpuHcHead(module))
                logger.info("[MHCPostConverter] [HcHead forward] Applied.")


@register_model_converter("npu_mhc_post")
class MHCPostModelConfig(ModelCustomConfig):
    model_converter = MHCPostConverter
