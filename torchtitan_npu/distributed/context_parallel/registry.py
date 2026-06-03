# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plugin registry for custom Context Parallel strategies.

CP strategies are registered via ``register_cp_strategy(detector, applier)``.
When ``apply_cp_to_attention_module`` walks the model, each module is tested
against registered detectors.  The first matching strategy is applied.
Unmatched modules raise ``NotImplementedError``.
"""

from collections.abc import Callable, Sequence

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.tools.logging import logger

_cp_strategies: list[
    tuple[Callable[[nn.Module], bool], Callable[[nn.Module, DeviceMesh], None]]
] = []


def register_cp_strategy(
    detector: Callable[[nn.Module], bool],
    applier: Callable[[nn.Module, DeviceMesh], None],
) -> None:
    """Register a CP strategy.

    Args:
        detector: Receives a module, returns True if this strategy applies.
        applier: Receives (module, cp_mesh), applies the CP parallelization.
    """
    _cp_strategies.append((detector, applier))


def apply_cp_to_attention_module(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
) -> None:
    """Apply custom CP strategies to each attention module.

    Each module is tested against registered detectors.  The first matching
    strategy is applied.  Unmatched modules raise ``NotImplementedError``.

    Args:
        attention_modules: Sequence of attention modules to apply CP to.
        cp_mesh: Device mesh for the context parallel dimension.
    """
    for module in attention_modules:
        module_name = type(module).__name__
        applied = False
        for detector, applier in _cp_strategies:
            if detector(module):
                logger.info(
                    f"CP: matched strategy for {module_name} "
                    f"cp_degree={cp_mesh.size()}"
                )
                applier(module, cp_mesh)
                applied = True
                break
        if not applied:
            raise NotImplementedError(
                f"No custom CP strategy found for module "
                f"{module_name}.  Registered detectors have been exhausted."
            )

    logger.info("Applied custom Context Parallel to the model")
