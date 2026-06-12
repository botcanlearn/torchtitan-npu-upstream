# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import parallel
from torch.distributed.tensor.parallel.style import ParallelStyle

if TYPE_CHECKING:
    from torchtitan_npu.converters.model_custom_interface import ParallelizePlanUpdater

logger = logging.getLogger(__name__)

_torch_parallelize_module = parallel.parallelize_module

_updater_cls_list: list[type["ParallelizePlanUpdater"]] = []


def parallelize_module_wrapper(
    module: nn.Module,
    device_mesh: DeviceMesh | None = None,
    parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None = None,
) -> nn.Module:
    """Create a wrapped parallelize_module function"""

    for _updater_cls_ in _updater_cls_list:
        parallelize_plan = _updater_cls_.update(parallelize_plan)

    return _torch_parallelize_module(
        module=module,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan,
    )


# pyrefly: ignore [bad-assignment]
parallel.parallelize_module = parallelize_module_wrapper


def apply_parallelize_plan_update(updater_cls: type["ParallelizePlanUpdater"]):
    _updater_cls_list.append(updater_cls)

    logger.info(f"[apply_parallelize_plan_update] Add ParallelizePlanUpdater {updater_cls}.")
