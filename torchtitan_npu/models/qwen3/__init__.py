# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.qwen3 import (
    model_registry as _upstream_model_registry,
    Qwen3StateDictAdapter,
)

from torchtitan_npu.models.qwen3.parallelize import parallelize_qwen3


def model_registry(flavor: str):
    spec = _upstream_model_registry(flavor)
    return spec.__class__(
        name=spec.name,
        flavor=spec.flavor,
        model=spec.model,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=spec.pipelining_fn,
        build_loss_fn=spec.build_loss_fn,
        post_optimizer_build_fn=spec.post_optimizer_build_fn,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
