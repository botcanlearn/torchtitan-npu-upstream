# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "PADDED_PATCH_COORDINATE",
    "DenseMaskSDPA",
    "build_causal_mask",
    "build_config",
    "build_document_ids",
    "build_encoder_causal_mask",
    "build_encoder_full_mask",
    "build_text_document_causal_mask",
    "build_valid_patch_mask",
    "config_to_dict",
    "require_config",
    "scatter_visual_embeddings",
]

from .attention import DenseMaskSDPA
from .masks import (
    PADDED_PATCH_COORDINATE,
    build_causal_mask,
    build_document_ids,
    build_encoder_causal_mask,
    build_encoder_full_mask,
    build_text_document_causal_mask,
    build_valid_patch_mask,
)
from .scatter import scatter_visual_embeddings
from .utils import build_config, config_to_dict, require_config
