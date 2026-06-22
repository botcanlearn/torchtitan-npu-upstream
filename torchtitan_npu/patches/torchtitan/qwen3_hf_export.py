# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Patch Qwen3StateDictAdapter.to_hf to keep fqn_to_index_mapping in sync.

When enable_weight_tying=True, to_hf skips output.weight (omitting
lm_head.weight from the HF state dict) but does not remove lm_head.weight
from fqn_to_index_mapping.  This mismatch causes the HuggingFace safetensors
consolidator to create an output file for lm_head.weight with empty metadata,
producing a corrupt 78-byte safetensors file.

This patch wraps to_hf to filter fqn_to_index_mapping after conversion.
"""

from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.tools.logging import logger

_ORIGINAL_TO_HF = Qwen3StateDictAdapter.to_hf


def _patched_to_hf(self, state_dict):
    hf_state_dict = _ORIGINAL_TO_HF(self, state_dict)

    # Remove entries not present in the actual to_hf output
    if self.fqn_to_index_mapping is not None:
        self.fqn_to_index_mapping = {k: v for k, v in self.fqn_to_index_mapping.items() if k in hf_state_dict}

    return hf_state_dict


Qwen3StateDictAdapter.to_hf = _patched_to_hf
logger.info("[Patch] Qwen3StateDictAdapter.to_hf: fqn_to_index_mapping sync enabled")
