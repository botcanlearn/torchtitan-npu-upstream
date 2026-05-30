# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "apply_parallelize_plan_update",
    "apply_state_dict_update",
]

from .parallelize_plan_update_wrapper import apply_parallelize_plan_update
from .state_dict_update_wrapper import apply_state_dict_update
