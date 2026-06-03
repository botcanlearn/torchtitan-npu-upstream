# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.models.deepseek_v3.parallelize as titan_deepseekv3_parallelize
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)

from torchtitan.distributed import ParallelDims
from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.protocols import ModelConvertersContainer


def parallelize_deepseekv3(
    model: DeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    _upstream_kwargs = dict(
        parallel_dims=parallel_dims,
        training=training,
        model_converters=model_converters,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )

    if parallel_dims.cp_enabled:
        cp_degree = parallel_dims.cp
        first_layer = next(iter(model.layers.values()))
        n_heads = first_layer.attention.n_heads  # pyrefly: ignore [missing-attribute]
        if n_heads % cp_degree != 0:
            raise ValueError(
                f"[Ulysses CP] n_heads={n_heads} must be divisible by "
                f"context_parallel_degree={cp_degree}."
            )

        from torchtitan_npu.distributed.context_parallel.registry import (
            apply_cp_to_attention_module as apply_cp,
        )

        titan_deepseekv3_parallelize.apply_cp_to_attention_module = apply_cp

    return titan_deepseekv3_parallelize.parallelize_deepseekv3(
        model, **_upstream_kwargs
    )
