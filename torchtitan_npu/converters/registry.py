# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import Configurable

from .framework.model_custom_config_registry import registry as npu_registry


def registry():
    return npu_registry


def register_model_converter(name: str):
    return npu_registry.register(name)


def get_model_converter_config(name: str) -> Configurable.Config | None:
    return npu_registry.get_config(name)


def has_npu_converter(converters: list, name: str) -> bool:
    """Return True if ``converters`` contains an NPU converter Config registered under ``name``.

    The dynamically-generated converter class carries ``_model_config``
    whose ``.name`` is the registered patch name; its ``Config`` carries
    ``_owner`` pointing back at that converter class.

    ``converters`` contains Config instances; hop through ``_owner`` to read
    the name. Also accept the raw converter class on the off chance the list
    ever contains them.
    """
    for c in converters:
        owner = getattr(c, "_owner", None) or c
        model_config = getattr(owner, "_model_config", None)
        if model_config is not None and getattr(model_config, "name", None) == name:
            return True
    return False
