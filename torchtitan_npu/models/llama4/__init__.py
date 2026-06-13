# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama4 import model_registry as _upstream_model_registry


def model_registry(flavor: str, **kwargs):
    return _upstream_model_registry(flavor, **kwargs)
