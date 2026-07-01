# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.models.qwen3.parallelize as titan_qwen3_parallelize
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.qwen3 import Qwen3Model
from torchtitan.protocols import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_qwen3(
    model: Qwen3Model,
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
                f"[Ulysses CP] n_heads={n_heads} must be divisible by context_parallel_degree={cp_degree}."
            )
        n_kv_heads = first_layer.attention.n_kv_heads  # pyrefly: ignore [missing-attribute]
        if n_kv_heads % cp_degree != 0:
            raise ValueError(
                f"[Ulysses CP] n_kv_heads={n_kv_heads} must be divisible by context_parallel_degree={cp_degree}."
            )

        logger.info(f"[Ulysses CP] Qwen3 Ulysses CP enabled: cp_degree={cp_degree}, n_heads={n_heads}")

        from torchtitan_npu.distributed.context_parallel.registry import (
            apply_cp_to_attention_module as apply_cp,
        )

        titan_qwen3_parallelize.apply_cp_to_attention_module = apply_cp

    return titan_qwen3_parallelize.parallelize_qwen3(model, **_upstream_kwargs)
