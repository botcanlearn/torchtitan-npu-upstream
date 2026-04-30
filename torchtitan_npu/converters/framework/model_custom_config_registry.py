# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.converters.model_custom_interface import ModelCustomConfig


class ConverterRegistry:
    _instance = None
    _model_configs: dict[str, ModelCustomConfig]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_configs = {}
        return cls._instance

    @staticmethod
    def _register_as_model_converter(
        name: str,
        config: ModelCustomConfig,
    ):
        from torchtitan.protocols.model_converter import register_model_converter

        from .model_custom_config_converter import ModelCustomConfigConverter

        converter_cls = type(
            f"{name}ModelCustomConfigConverter",
            (ModelCustomConfigConverter,),
            {
                "_model_config": config,
            },
        )

        register_model_converter(converter_cls, name)

    def register(
        self,
        name: str,
    ):
        def decorator(config: ModelCustomConfig):
            config.name = name
            self._model_configs[name] = config
            self._register_as_model_converter(name, config)
            return config

        return decorator

    def get(self, name: str) -> ModelCustomConfig | None:
        return self._model_configs.get(name)
