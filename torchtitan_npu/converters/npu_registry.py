# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.protocols.train_spec as train_spec_module

from .framework.model_custom_config_registry import ConverterRegistry

registry = ConverterRegistry()


def register_model_converter(name: str):
    return registry.register(name)


G_CUR_USING_TRAIN_SPEC: train_spec_module.TrainSpec | None = None


def get_using_train_spec() -> train_spec_module.TrainSpec | None:
    return G_CUR_USING_TRAIN_SPEC


_original_get_train_spec = train_spec_module.get_train_spec


def get_train_spec_wrapper(name: str) -> train_spec_module.TrainSpec:
    train_spec = _original_get_train_spec(name)
    global G_CUR_USING_TRAIN_SPEC
    G_CUR_USING_TRAIN_SPEC = train_spec

    return train_spec


train_spec_module.get_train_spec = get_train_spec_wrapper
