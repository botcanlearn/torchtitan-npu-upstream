# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan_npu.converters.quant_converter import module_filter_fn


def test_module_filter_fn_accepts_unfiltered_linear():
    mod = nn.Linear(8, 16)

    assert module_filter_fn(mod, "model.layers.0.mlp.w1", ["attention"])


def test_module_filter_fn_rejects_non_linear_module():
    mod = nn.ReLU()

    assert not module_filter_fn(mod, "model.layers.0.relu", [])


def test_module_filter_fn_rejects_filtered_fqn():
    mod = nn.Linear(8, 16)

    assert not module_filter_fn(mod, "model.layers.0.attention.proj", ["attention"])
