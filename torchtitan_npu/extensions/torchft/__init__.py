# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""NPU extensions to the TorchTitan TorchFT trainer."""

# Load backports before the trainer imports upstream classes and Config factories.
from torchtitan_npu.experiments import torchft as _torchft  # noqa: F401
