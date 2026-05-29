# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from NVIDIA Megatron-LM,
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/transformer/experimental_attention_variant/dsa.py
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.protocols.module import Module

logger = logging.getLogger(__name__)

LOSS_SCALE = torch.tensor(1.0)


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for DSA indexer loss."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The indexer loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                                               gradient.
        """
        (loss,) = ctx.saved_tensors
        LOSS_SCALE.to(device=loss.device)
        scaled_dsa_indexer_loss_grad = torch.ones_like(loss) * LOSS_SCALE
        return grad_output, scaled_dsa_indexer_loss_grad

    @classmethod
    def set_loss_scale(cls, scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        global LOSS_SCALE
        LOSS_SCALE = scale


class DSAIndexerLossLoggingHelper:
    """Helper class for logging DSAIndexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
    ):
        """Save the DSA indexer loss for logging.
        Args:
            loss (torch.Tensor): The loss tensor.
            layer_number (int): 0-based layer index of the loss.
            num_layers (int): The number of total layers.
        """
        # Skip DSA indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=loss.device)

        layer_number = int(layer_number)
        if not 0 <= layer_number < num_layers:
            raise ValueError(
                f"DSA indexer layer_number must be in [0, {num_layers}), "
                f"got {layer_number}."
            )

        tracker["values"][layer_number] += (
            loss.to_local().detach() if isinstance(loss, DTensor) else loss.detach()
        )

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the DSA indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        values = tracker.get("values")
        if values is not None:
            values.zero_()

    @staticmethod
    def track_dsa_indexer_metrics(total_acc_steps: int):
        """Track the DSA Indexer metrics for logging."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        das_indexer_losses = tracker["values"]
        valid = das_indexer_losses != 0
        if torch.count_nonzero(valid).item() == 0:
            DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
            return
        loss = (das_indexer_losses[valid] / total_acc_steps).mean()
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        logger.info(f"indexer loss: {loss.item()}")


class DSAIndexerLoss(Module):
    """Compute dsa indexer loss at sparse training stage.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        eps: float = 1e-8

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.eps = config.eps

    def forward(
        self,
        selected_main_attn_dist,
        index_score,
        topk_indices,
        loss_scale,
    ):
        index_score = F.softmax(index_score, dim=-1, dtype=torch.float32)

        # considering only the selected token
        selected_main_attn_dist = F.normalize(selected_main_attn_dist, p=1, dim=-1)
        loss = (
            F.kl_div(
                (index_score + self.eps).log(),
                selected_main_attn_dist + self.eps,
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
        )
        loss *= loss_scale
        return loss
