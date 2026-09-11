# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TileLang fused DeepSeek-V4 kernels exposed as torch custom operators.

The kernels are registered via ``torch.library.custom_op``. Their fake
implementations keep ``torch.compile``/AOT eager graph capture independent of
the optional TileLang runtime package.
"""

__all__ = ["mhc_head_compute_mix_tilelang", "tilelang_mhc_post", "tilelang_mhc_pre", "tilelang_swiglu"]

from .head_compute_mix import mhc_head_compute_mix_tilelang
from .mhc_post import tilelang_mhc_post
from .mhc_pre import tilelang_mhc_pre
from .swiglu import tilelang_swiglu
