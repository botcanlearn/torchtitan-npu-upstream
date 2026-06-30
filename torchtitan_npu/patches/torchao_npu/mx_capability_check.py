# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan.tools.utils.has_cuda_capability

Target:
- torchtitan.tools.utils.has_cuda_capability

Reason:
- Upstream's MXFP8Converter.__init__ (torchtitan/components/quantization/mx.py)
  calls has_cuda_capability(10, 0) to guard MXFP8 behind Hopper+ (SM100) GPU
  hardware.  On Ascend NPU there is no CUDA device, so the check always fails
  even though torchao + torchtitan_npu provide NPU-native MXFP8 kernels.

- This patch replaces has_cuda_capability with an NPU-aware version that
  verifies the device is Ascend950 (the minimum NPU architecture supporting
  MXFP8) and returns True accordingly, allowing MXFP8Converter to initialise.

Idempotency:
- Importing this module installs the wrapper exactly once thanks to Python's
  module cache.
"""

from torchtitan.tools import utils as tt_utils

from torchtitan_npu.tools.device import get_npu_device_type


def has_mx_capability(major: int, minor: int) -> bool:
    """Replace has_cuda_capability with an Ascend950 architecture check.

    MXFP8 on NPU requires Ascend950 (or equivalent).  Returns True when the
    active device meets the requirement, otherwise raises RuntimeError with
    a clear message.
    """
    device_name = get_npu_device_type()
    if device_name == "A5":
        return True
    raise RuntimeError(f"MXFP8 is only supported on Ascend950 or higher architecture, but got device: {device_name}")


tt_utils.has_cuda_capability = has_mx_capability
