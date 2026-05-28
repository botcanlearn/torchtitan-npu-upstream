# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from typing import Any, TypeVar


ConfigT = TypeVar("ConfigT")


def config_to_dict(config) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def build_config(config_cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
    return config_cls(**values)


def require_config(
    config: object,
    config_cls: type[ConfigT],
    field_name: str,
) -> ConfigT:
    if not isinstance(config, config_cls):
        raise TypeError(
            f"Expected {field_name} to be {config_cls.__name__}, "
            f"got {type(config).__name__}."
        )
    return config
