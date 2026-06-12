# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torchtitan.config import ActivationCheckpointConfig, CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.vlm.infra.parallelize import parallelize_vlm
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import ParallelismConfig, TrainingConfig
from torchtitan_npu.converters.registry import has_npu_converter

# Current VLM NPU only supports FSDP/HSDP data parallelism.


def _validate_supported_parallel_dims(parallel_dims: ParallelDims) -> None:
    """Document and enforce the currently supported VLM NPU parallel modes."""
    if parallel_dims.tp_enabled or parallel_dims.pp_enabled or parallel_dims.cp_enabled:
        raise NotImplementedError("VLM NPU currently only supports FSDP/HSDP data parallelism.")


def _validate_npu_vlm_converter(
    model_converters: ModelConvertersContainer.Config,
) -> None:
    if not has_npu_converter(model_converters.converters, "npu_vlm"):
        raise ValueError('VLM NPU requires "npu_vlm" in model.converters.')


def parallelize_vlm_npu(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
) -> nn.Module:
    _validate_supported_parallel_dims(parallel_dims)
    _validate_npu_vlm_converter(model_converters)

    return parallelize_vlm(
        model,
        parallel_dims=parallel_dims,
        training=training,
        model_converters=model_converters,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
