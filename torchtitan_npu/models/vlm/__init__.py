# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["model_registry", "parallelize_vlm_npu"]

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.experiments.vlm import llama3_siglip2_configs
from torchtitan.protocols.model_spec import ModelSpec

from .model import to_npu_vlm_config
from .parallelize import parallelize_vlm_npu


def model_registry(flavor: str) -> ModelSpec:
    """Register VLM model with NPU-specific parallelization.

    Args:
        flavor: Model flavor name (e.g., 'debugmodel')

    Returns:
        ModelSpec with VLM configuration and NPU parallelization
    """
    config = to_npu_vlm_config(llama3_siglip2_configs[flavor]())
    return ModelSpec(
        name="vlm",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_vlm_npu,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
