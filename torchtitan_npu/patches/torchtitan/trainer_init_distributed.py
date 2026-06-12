# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Patch:
#   torchtitan.trainer.Trainer.init_distributed — intercept the call to
#   stash Trainer.Config on a module-level variable so that converter code
#   can read the active ModelSpec without receiving it through narrow
#   Config slices.
#
# Why:
#   ModelCustomConfigConverter needs the ModelSpec to determine which model
#   is being trained, but only receives narrow Config slices through its
#   `build()` call. This patch captures the full Trainer.Config (which
#   carries ModelSpec) at the earliest point in the training lifecycle.
#
# Safety:
#   The monkey-patch is NOT applied on import. It is applied by calling
#   ``apply()`` from ``_apply_patches()`` at the very start of the patch
#   sequence, ensuring we capture the true upstream ``init_distributed``
#   before any other patch can override it.

from collections.abc import Callable

from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

G_USING_TRAIN_CONFIG: Trainer.Config | None = None

_original_init_distributed: Callable | None = None


def get_using_model_spec() -> ModelSpec:
    if G_USING_TRAIN_CONFIG is None:
        raise RuntimeError("G_USING_TRAIN_CONFIG must be set before using.")
    return G_USING_TRAIN_CONFIG.model_spec  # pyrefly: ignore [bad-return]


def init_distributed_wrapper(self, *args, **kwargs):
    # *args/**kwargs: transparently pass through any parameter changes
    # upstream makes to init_distributed so this wrapper does not break
    # when the upstream signature evolves.
    global G_USING_TRAIN_CONFIG
    G_USING_TRAIN_CONFIG = self.config
    if _original_init_distributed is None:
        raise RuntimeError(
            "init_distributed_wrapper called before apply(); this patch must be activated via _apply_patches()."
        )
    return _original_init_distributed(self, *args, **kwargs)


def apply():
    """Install the ``init_distributed`` wrapper on ``Trainer``.

    Must be called from ``_apply_patches()`` before any other patch that
    might also modify ``Trainer.init_distributed``.
    """
    if not hasattr(Trainer, "init_distributed"):
        raise AttributeError(
            "torchtitan-npu: Trainer.init_distributed no longer exists. "
            "This patch needs to be updated to match the upstream torchtitan API. "
            "Check if the method was renamed, removed, or moved."
        )
    global _original_init_distributed
    _original_init_distributed = Trainer.init_distributed
    Trainer.init_distributed = init_distributed_wrapper
