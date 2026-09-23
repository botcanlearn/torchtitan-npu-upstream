# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility entry for the former TorchTitan EMA export backport.

Use torchtitan_npu.scripts.checkpoint_conversion.convert_to_hf for NPU exports.
Only this shim may be removed after callers migrate; the plugin implementation
owns NPU model discovery and Engram shard loading independently of upstream EMA.
"""

from torchtitan_npu.scripts.checkpoint_conversion.convert_to_hf import (
    ParallelFileSystemReader as ParallelFileSystemReader,
)
from torchtitan_npu.scripts.checkpoint_conversion.convert_to_hf import (
    _load_ema_state_dict as _load_ema_state_dict,
)
from torchtitan_npu.scripts.checkpoint_conversion.convert_to_hf import (
    convert_to_hf as convert_to_hf,
)
from torchtitan_npu.scripts.checkpoint_conversion.convert_to_hf import (
    main,
)

if __name__ == "__main__":
    main()
