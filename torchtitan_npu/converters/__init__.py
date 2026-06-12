# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

__all__ = [
    "ModelCustomConfig",
    "ParallelizePlanUpdater",
    "StateDictUpdater",
    "get_model_converter_config",
    "has_npu_converter",
    "register_model_converter",
    "registry",
]

import importlib
import pkgutil
from pathlib import Path

from .model_custom_interface import (
    ModelCustomConfig,
    ParallelizePlanUpdater,
    StateDictUpdater,
)
from .registry import (
    get_model_converter_config,
    has_npu_converter,
    register_model_converter,
    registry,
)


def _auto_search_conveter():
    package_dir = Path(__file__).parent

    for subdir in ["kernels", "features"]:
        subdir_path = package_dir / subdir
        if subdir_path.exists():
            for _, name, _ in pkgutil.iter_modules([str(subdir_path)]):
                importlib.import_module(f".{subdir}.{name}", package=__package__)


_auto_search_conveter()
