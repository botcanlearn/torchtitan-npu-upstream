# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4529
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep independent DSV4 metadata out of EP chunk concretization.

Filter placeholder extents by the selected chunk symbols before invoking the
upstream hint collector helper. Preserve unrelated symbolic values; delegate
concretization of selected values to the upstream implementation.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import torchtitan.experiments.graph_trainer.ep_pass_utils as ep_pass_utils
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    import torch.fx as fx


def _chunk_placeholder_symbol_hints(gm: fx.GraphModule) -> dict[object, int]:
    """Collect hints only from placeholder extents containing chunk symbols."""
    chunk_symbols = ep_pass_utils.chunk_symbol_hints_for_mode(gm).keys()
    hints: dict[object, int] = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder" or (val := ep_pass_utils.tensor_meta(node)) is None:
            continue
        for dim, extent in enumerate(val.shape):
            symbols = ep_pass_utils.free_symbols(extent)
            if not symbols & chunk_symbols:
                continue
            ep_pass_utils._record_symbols_from_extent(hints, extent, source=f"{node.name}.shape[{dim}]")
    return hints


def apply() -> None:
    required_helpers = (
        "_concretize_value",
        "_placeholder_symbol_hints",
        "_record_symbols_from_extent",
        "chunk_symbol_hints_for_mode",
        "free_symbols",
        "tensor_meta",
    )
    if not all(hasattr(ep_pass_utils, name) for name in required_helpers):
        logger.warning(
            "Skipping GraphTrainer EP chunk concretization patch: required symbolic-shape helpers are unavailable."
        )
        return

    current = ep_pass_utils._placeholder_symbol_hints
    if getattr(current, "npu_preserves_independent_symbols", False):
        return

    original_concretize_value = ep_pass_utils._concretize_value

    @wraps(original_concretize_value)
    def _concretize_chunk_value(value: object, symbol_hints: dict[object, int]) -> object:
        if not ep_pass_utils.free_symbols(value) & symbol_hints.keys():
            return value
        return original_concretize_value(value, symbol_hints)

    # pyrefly: ignore [missing-attribute]
    _chunk_placeholder_symbol_hints.npu_preserves_independent_symbols = True
    ep_pass_utils._placeholder_symbol_hints = _chunk_placeholder_symbol_hints
    ep_pass_utils._concretize_value = _concretize_chunk_value
    logger.info("Enabled GraphTrainer EP chunk concretization patch for independent dynamic shapes")


apply()
