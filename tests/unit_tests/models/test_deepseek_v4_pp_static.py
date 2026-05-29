# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
DSV4_DIR = REPO_ROOT / "torchtitan_npu" / "models" / "deepseek_v4"


def _make_pipeline_import_stubs() -> dict[str, types.ModuleType]:
    torch_mod = types.ModuleType("torch")
    torch_mod.device = type("device", (), {})

    dist_mod = types.ModuleType("torch.distributed")
    pipelining_mod = types.ModuleType("torch.distributed.pipelining")
    schedules_mod = types.ModuleType("torch.distributed.pipelining.schedules")

    class PipelineScheduleSingle:
        pass

    class PipelineScheduleMulti:
        pass

    def get_schedule_class(name):
        return PipelineScheduleSingle if name == "1F1B" else PipelineScheduleMulti

    schedules_mod.PipelineScheduleSingle = PipelineScheduleSingle
    schedules_mod.get_schedule_class = get_schedule_class
    pipelining_mod.schedules = schedules_mod
    dist_mod.pipelining = pipelining_mod
    torch_mod.distributed = dist_mod

    torchtitan_mod = types.ModuleType("torchtitan")
    titan_dist_mod = types.ModuleType("torchtitan.distributed")
    titan_pp_mod = types.ModuleType("torchtitan.distributed.pipeline_parallel")

    def build_pipeline_schedule_stub(*args, **kwargs):
        return None

    def pipeline_module_split_stub(*args, **kwargs):
        return [], []

    def generate_llm_fqn_per_model_part_stub(
        num_stages, num_layers, input_weight=1, output_weight=1
    ):
        # Faithful (torch-free) reimplementation of the upstream generator so the
        # hermetic static test can exercise generate_deepseek_v4_fqn_per_model_part
        # without importing the real torchtitan/torch stack.
        if num_stages < 1:
            raise ValueError("Number of stages must be at least 1")
        if num_stages == 1:
            layer_names = [f"layers.{i}" for i in range(num_layers)]
            return [["tok_embeddings"] + layer_names + ["norm", "output"]]

        num_effective_layers = num_layers + input_weight + output_weight
        if num_stages > num_effective_layers:
            raise ValueError("Number of stages cannot exceed effective layers")
        layers_per_stage = num_effective_layers // num_stages
        extra_layers = num_effective_layers % num_stages

        module_names_per_stage = []
        current_layer = 0
        for stage_idx in range(num_stages):
            stage_modules = []
            effective = layers_per_stage + (1 if stage_idx < extra_layers else 0)
            if stage_idx == 0:
                stage_modules.append("tok_embeddings")
                effective -= input_weight
            elif stage_idx == num_stages - 1:
                effective -= output_weight
            for _ in range(effective):
                if current_layer < num_layers:
                    stage_modules.append(f"layers.{current_layer}")
                    current_layer += 1
            if stage_idx == num_stages - 1:
                stage_modules.extend(["norm", "output"])
            module_names_per_stage.append(stage_modules)
        return module_names_per_stage

    def logger_noop(*args, **kwargs):
        return None

    titan_pp_mod.build_pipeline_schedule = build_pipeline_schedule_stub
    titan_pp_mod.pipeline_module_split = pipeline_module_split_stub
    titan_pp_mod.generate_llm_fqn_per_model_part = generate_llm_fqn_per_model_part_stub
    titan_tools_mod = types.ModuleType("torchtitan.tools")
    titan_logging_mod = types.ModuleType("torchtitan.tools.logging")
    titan_logging_mod.logger = types.SimpleNamespace(
        debug=logger_noop,
        info=logger_noop,
    )

    return {
        "torch": torch_mod,
        "torch.distributed": dist_mod,
        "torch.distributed.pipelining": pipelining_mod,
        "torch.distributed.pipelining.schedules": schedules_mod,
        "torchtitan": torchtitan_mod,
        "torchtitan.distributed": titan_dist_mod,
        "torchtitan.distributed.pipeline_parallel": titan_pp_mod,
        "torchtitan.tools": titan_tools_mod,
        "torchtitan.tools.logging": titan_logging_mod,
    }


def _load_pipeline_module():
    path = DSV4_DIR / "pipeline_parallel.py"
    assert path.exists(), "DeepSeek V4 must define a dedicated PP pipeline module"
    with mock.patch.dict(sys.modules, _make_pipeline_import_stubs()):
        spec = importlib.util.spec_from_file_location(
            "deepseek_v4_pipeline_parallel", path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


def _class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"{class_name} not found in {path}")


class DeepSeekV4PipelineStaticTest(unittest.TestCase):
    def test_pipeline_import_stubs_do_not_leak(self):
        watched_modules = _make_pipeline_import_stubs().keys()
        missing = object()
        before = {name: sys.modules.get(name, missing) for name in watched_modules}

        _load_pipeline_module()

        for name, original_module in before.items():
            if original_module is missing:
                self.assertNotIn(name, sys.modules)
            else:
                self.assertIs(sys.modules[name], original_module)

    def test_generator_keeps_hc_head_on_last_stage(self):
        module = _load_pipeline_module()

        stages = module.generate_deepseek_v4_fqn_per_model_part(
            num_stages=4,
            num_layers=4,
            input_weight=1,
            output_weight=1,
        )

        self.assertEqual(
            stages,
            [
                ["tok_embeddings", "layers.0"],
                ["layers.1", "layers.2"],
                ["layers.3"],
                ["hc_head", "norm", "output"],
            ],
        )
        module.validate_deepseek_v4_stage_modules(stages, 4, 4)

    def test_generator_validates_stage_invariants(self):
        module = _load_pipeline_module()

        with self.assertRaisesRegex(ValueError, "output modules"):
            module.validate_deepseek_v4_stage_modules(
                [["tok_embeddings", "hc_head"], ["layers.0", "norm", "output"]],
                1,
                2,
            )
        with self.assertRaisesRegex(ValueError, "tok_embeddings"):
            module.validate_deepseek_v4_stage_modules(
                [["layers.0"], ["tok_embeddings", "hc_head", "norm", "output"]],
                1,
                2,
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            module.validate_deepseek_v4_stage_modules(
                [["tok_embeddings"], ["hc_head", "norm", "output"]],
                1,
                2,
            )

    def test_pipeline_fn_matches_current_torchtitan_signature(self):
        module = _load_pipeline_module()
        signature = inspect.signature(module.pipeline_deepseek_v4)

        expected_params = [
            "model",
            "parallel_dims",
            "training",
            "model_converters",
            "parallelism",
            "compile_config",
            "ac_config",
            "dump_folder",
            "device",
            "model_config",
            "parallelize_fn",
            "loss_fn",
        ]
        self.assertEqual(list(signature.parameters), expected_params)
        for name in expected_params[1:]:
            self.assertEqual(
                signature.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
                name,
            )

        source = (DSV4_DIR / "pipeline_parallel.py").read_text()
        self.assertIn("build_pipeline_schedule(", source)
        self.assertIn("parallelism=parallelism", source)
        self.assertIn("local_batch_size=training.local_batch_size", source)
        self.assertNotIn("build_pipeline_schedule(config, stages, loss_fn)", source)

    def test_model_forward_declares_pp_sidecar_protocol(self):
        path = DSV4_DIR / "model.py"
        source = path.read_text()
        methods = _class_methods(path, "DeepSeekV4Model")

        self.assertIn("_normalize_pp_input_ids", methods)
        self.assertIn("_validate_last_stage_hc_head", methods)
        self.assertIn("input_ids = self._normalize_pp_input_ids(input_ids)", source)
        self.assertIn("if self.output is None:", source)
        self.assertIn("return h", source)
        self.assertIn("layer_id = cast(Any, layer).layer_id", source)
        self.assertIn("if layer_id < self.model_args.n_layers", source)
        self.assertIn("self._validate_last_stage_hc_head()", source)

    def test_parallelize_uses_dynamic_root_plan_for_pp_chunks(self):
        source = (DSV4_DIR / "parallelize.py").read_text()

        self.assertIn("root_parallelize_plan: dict[str, Any] = {}", source)
        self.assertIn('tok_embeddings = getattr(model, "tok_embeddings", None)', source)
        self.assertIn('hc_head = getattr(model, "hc_head", None)', source)
        self.assertIn("if tok_embeddings is not None:", source)
        self.assertIn("if hc_head is not None:", source)
        self.assertIn("hc_head_plan = prepare_module_input_output(", source)
        self.assertIn("use_local_input=False", source)
        self.assertIn("apply_distributed_indexer_loss_tracking(", source)
        self.assertIn("model_args = cast(Any, model).model_args", source)
        self.assertIn(
            "parallel_dims, model_args.n_layers, model_args.compress_ratios",
            source,
        )

    def test_pipelining_patch_injects_deepseek_v4_input_ids(self):
        patch_source = (
            REPO_ROOT / "torchtitan_npu" / "patches" / "torch" / "pipelining.py"
        ).read_text()
        pipeline_source = (DSV4_DIR / "pipeline_parallel.py").read_text()

        self.assertIn(
            "_patch_post_dataloading_process_for_deepseek_v4_pp_input_ids",
            patch_source,
        )
        self.assertIn(
            "from torchtitan_npu.models.deepseek_v4.pipeline_parallel import",
            patch_source,
        )
        self.assertNotIn("def _with_deepseek_v4_pp_input_ids", patch_source)
        self.assertIn("def _is_deepseek_v4_pp_target", pipeline_source)
        self.assertIn("def _with_deepseek_v4_pp_input_ids", pipeline_source)
        self.assertIn(
            'extra_kwargs["input_ids"] = input_ids.detach().long()',
            pipeline_source,
        )
        self.assertIn(
            'if devices is not None and "device_type" not in kwargs',
            patch_source,
        )
        self.assertIn('kwargs["device_type"] = "npu"', patch_source)

    def test_dsa_tracker_records_zero_based_layers_and_distributed_reduce(self):
        common_source = (
            REPO_ROOT / "torchtitan_npu" / "models" / "common" / "dsa_indexer_loss.py"
        ).read_text()
        parallelize_source = (DSV4_DIR / "parallelize.py").read_text()

        self.assertIn('tracker["values"][layer_number]', common_source)
        self.assertIn("valid = das_indexer_losses != 0", common_source)
        self.assertNotIn('tracker["present"]', common_source)
        self.assertIn("valid_indices = [", parallelize_source)
        self.assertIn("compress_ratios[i] == 4", parallelize_source)
        self.assertIn("dist.all_reduce(dsa_indexer_losses", parallelize_source)
        self.assertIn(
            "norm_factor = dist.get_world_size() // max(parallel_dims.pp, 1)",
            parallelize_source,
        )


if __name__ == "__main__":
    unittest.main()
