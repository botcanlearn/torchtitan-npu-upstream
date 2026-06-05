# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

_NPU_DEVICE_TYPE_MAP = {
    "Ascend950DT": "A5",
    "Ascend950PR": "A5",
    "Ascend910_95": "A5",
    "Ascend950": "A5",
    "Ascend910_93": "A3",
    "Ascend910B": "A2",
}


def _get_npu_device_name() -> str:
    try:
        import torch_npu

        return torch_npu.npu.get_device_name()
    except Exception:
        return ""


def get_npu_device_type() -> str:
    device_name = _get_npu_device_name()
    for marker, device_type in _NPU_DEVICE_TYPE_MAP.items():
        if marker in device_name:
            return device_type
    return "UNKNOWN"
