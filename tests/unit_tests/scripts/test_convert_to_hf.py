# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file


@pytest.fixture(autouse=True)
def cleanup_test_pycache():
    yield
    pycache_dir = Path(__file__).with_name("__pycache__")
    for cache_file in pycache_dir.glob(f"{Path(__file__).stem}*.pyc"):
        cache_file.unlink()
    try:
        pycache_dir.rmdir()
    except OSError:
        pass


def _install_fake_script_imports(monkeypatch):
    class ModelWrapper:
        def __init__(self, model):
            self.model = model

        def _get_state_dict(self):
            return self.model.state_dict()

    def package(name):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    package("torchtitan")
    package("torchtitan.components")
    checkpoint_module = types.ModuleType("torchtitan.components.checkpoint")
    checkpoint_module.ModelWrapper = ModelWrapper
    monkeypatch.setitem(
        sys.modules, "torchtitan.components.checkpoint", checkpoint_module
    )

    config_module = types.ModuleType("torchtitan.config")
    config_module.TORCH_DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    monkeypatch.setitem(sys.modules, "torchtitan.config", config_module)

    package("torchtitan_npu")
    package("torchtitan_npu.converters")
    package("torchtitan_npu.converters.framework")
    state_dict_update_module = types.ModuleType(
        "torchtitan_npu.converters.framework.state_dict_update_wrapper"
    )

    def _fake_apply_state_dict_update(updater_cls, model_spec):
        return None

    state_dict_update_module.apply_state_dict_update = _fake_apply_state_dict_update
    monkeypatch.setitem(
        sys.modules,
        "torchtitan_npu.converters.framework.state_dict_update_wrapper",
        state_dict_update_module,
    )

    package("torchtitan_npu.converters.kernels")
    gmm_module = types.ModuleType("torchtitan_npu.converters.kernels.gmm")
    gmm_module.GMMStateDictUpdater = object

    def _fake_convert(model):
        return None

    def _fake_npu_grouped_expert_converter(model_spec):
        return types.SimpleNamespace(convert=_fake_convert)

    gmm_module.NpuGroupedExpertConverter = _fake_npu_grouped_expert_converter
    monkeypatch.setitem(
        sys.modules, "torchtitan_npu.converters.kernels.gmm", gmm_module
    )


def _load_convert_to_hf_module(monkeypatch):
    _install_fake_script_imports(monkeypatch)
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "checkpoint_conversion" / "convert_to_hf.py"
    spec = importlib.util.spec_from_file_location(
        "convert_to_hf_under_test", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyConfig:
    @staticmethod
    def build():
        model = torch.nn.Linear(3, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        return model


class TinyStateDictAdapter:
    def __init__(self, model_config, hf_assets_path):
        self.fqn_to_index_mapping = {
            "hf_model.linear.weight": 1,
            "hf_model.linear.bias": 1,
        }

    @staticmethod
    def to_hf(state_dict):
        return {
            "hf_model.linear.weight": state_dict["weight"],
            "hf_model.linear.bias": state_dict["bias"],
        }


def test_convert_to_hf_writes_cpu_dcp_checkpoint_as_hf_safetensors(
    monkeypatch, tmp_path
):
    convert_to_hf = _load_convert_to_hf_module(monkeypatch)

    model_spec = types.SimpleNamespace(
        model=TinyConfig(),
        state_dict_adapter=TinyStateDictAdapter,
    )
    model_module = types.SimpleNamespace(model_registry=lambda flavor: model_spec)
    monkeypatch.setattr(
        convert_to_hf, "_import_model_module", lambda model_name: model_module
    )

    expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expected_bias = torch.tensor([3.5, -2.5], dtype=torch.float32)
    input_dir = tmp_path / "input_dcp"
    output_dir = tmp_path / "output_hf"
    dcp.save(
        {"weight": expected_weight, "bias": expected_bias},
        checkpoint_id=input_dir,
    )

    convert_to_hf.convert_to_hf(
        input_dir=input_dir,
        output_dir=output_dir,
        model_name="tiny",
        model_flavor="debug",
        hf_assets_path=tmp_path / "hf_assets",
        export_dtype="float32",
    )

    safetensors_path = output_dir / "model-00001-of-00001.safetensors"
    saved_tensors = load_file(str(safetensors_path))

    assert torch.equal(saved_tensors["hf_model.linear.weight"], expected_weight)
    assert torch.equal(saved_tensors["hf_model.linear.bias"], expected_bias)
