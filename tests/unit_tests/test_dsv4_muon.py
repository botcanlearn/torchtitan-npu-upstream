# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
from pathlib import Path

from torchtitan.distributed.flex_shard import BlockShard

from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.models.deepseek_v4.config_registry import _dsv4_muon_profile

_ROOT = Path(__file__).resolve().parents[2]


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(path.read_text(), node) or ""
    raise AssertionError(f"function {name!r} not found in {path}")


def test_dist_muon_patch_admits_only_npu_storage():
    path = _ROOT / "torchtitan_npu/patches/torchtitan/distributed/flex_shard/dist_muon.py"
    source = _function_source(path, "_validate_parameter_storage")

    assert "isinstance(param, DTensor)" in source
    assert 'local_device.type != "npu"' in source
    assert "requires NPU parameters" in source
    assert "len(local_devices) != 1" in source
    assert "requires one device per process" in source


def test_d2h_swap_cleanup_warns_before_releasing_the_pinned_buffer():
    path = _ROOT / "torchtitan_npu/extensions/novaswap/swap_primitive.py"
    source = _function_source(path, "d2h")

    assert "logger.warning" in source
    assert "D2H swap submission failed" in source
    assert "stream.synchronize()" in source
    assert "PinnedCpuStorage.free(tensor_cpu, owner=owner)" in source
    assert "raise" in source


def test_dsv4_muon_profile_uses_validated_phase_one_policy():
    path = _ROOT / "torchtitan_npu/models/deepseek_v4/config_registry.py"
    file_source = path.read_text()
    source = _function_source(path, "_dsv4_muon_profile")

    assert '"wq_a": owned' in source
    assert '"wkv": owned' in source
    assert '"wo_b": owned' in source
    assert "shared_experts.{projection}.weight" in source
    optimizer_source = _function_source(path, "_dsv4_optimizer_config")
    assert "muon_momentum=0.95" in optimizer_source
    assert "muon_ns_steps=10" in optimizer_source
    assert "beta1=0.9" in optimizer_source
    assert "eps=1e-8" in optimizer_source
    trainer_source = (_ROOT / "torchtitan_npu/extensions/trainer.py").read_text()
    assert "tensor_parallel_degree=1" in trainer_source
    assert "pipeline_parallel_degree=1" in trainer_source
    assert "def deepseek_v4_debugmodel_muon" not in file_source
    assert "def deepseek_v4_flash_muon" not in file_source
    assert "def deepseek_v4_flash_43layers_16experts_muon" not in file_source
    assert "def deepseek_v4_pro_muon" not in file_source
    # The DSV4 profile is the single unified active Muon policy.
    assert "include_mtp_projections" in source
    assert 'f"mtp_layers.{layer_id}"' in source
    assert 'mtp_projections = ("e_proj", "h_proj")' in source
    assert "compressor.ape" in source
    assert "deepseek_v4_flash_43layers_16experts_muon_indexer" not in file_source
    for experimental_recipe in (
        "deepseek_v4_flash_43layers_16experts_muon_attention",
        "deepseek_v4_flash_43layers_16experts_muon_compressor",
        "deepseek_v4_flash_43layers_16experts_muon_routed_experts",
        "deepseek_v4_flash_43layers_16experts_muon_router",
        "deepseek_v4_flash_43layers_16experts_muon_mhc",
    ):
        assert experimental_recipe not in file_source


def test_dsv4_unified_muon_policy_follows_paper_parameter_split():
    path = _ROOT / "torchtitan_npu/models/deepseek_v4/config_registry.py"
    source = path.read_text()
    active_source = _function_source(path, "_dsv4_muon_profile")

    assert "mtp_layers" in active_source
    assert "_PAPER_" not in source
    assert "_distributed_paper_parameter_muon_optimizer" not in source
    assert "deepseek_v4_debugmodel_paper_muon" not in source
    assert "BlockShard(dim=0" in source
    assert "MeshAxisName.DP_SHARD.value: Shard(0)" in source
    assert "MeshAxisName.EFSDP.value: Shard(0)" in source
    for unified_muon_parameter in (
        "indexer",
        "compressor",
        "routed_experts",
        "router",
        'hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")',
        'mtp_projections = ("e_proj", "h_proj")',
    ):
        assert unified_muon_parameter in active_source


def test_dsv4_muon_indexer_query_uses_per_head_block_sharding():
    model_spec = model_registry("debugmodel")
    profile = _dsv4_muon_profile(model_spec)
    compute_shardings = profile.optimizer_factory_kwargs["DistMuon"][
        "compute_sharding_by_fqn"
    ]

    main_query_layout = compute_shardings["layers.2.attention.wq_b.weight"]
    indexer_query_layout = compute_shardings[
        "layers.2.attention.indexer.wq_b.weight"
    ]

    main_dp_sharding = main_query_layout.shardings_by_mesh_axis["dp_shard"]
    assert isinstance(main_dp_sharding, BlockShard)
    assert main_dp_sharding.dim == 0
    assert main_dp_sharding.block_size == model_spec.model.layers[2].attention.head_dim

    indexer_dp_sharding = indexer_query_layout.shardings_by_mesh_axis["dp_shard"]
    assert isinstance(indexer_dp_sharding, BlockShard)
    assert indexer_dp_sharding.dim == 0
    indexer = model_spec.model.layers[2].attention.indexer
    assert indexer is not None
    assert indexer_dp_sharding.block_size == indexer.index_head_dim


def test_graph_trainer_conversion_strips_only_npu_top_level_extension():
    registry_path = _ROOT / "torchtitan_npu/models/deepseek_v4/config_registry.py"
    conversion_source = _function_source(registry_path, "to_graph_trainer_config")

    assert "isinstance(base_config, TrainerEx.Config)" in conversion_source
    assert "for config_field in fields(Trainer.Config)" in conversion_source

    for graph_recipe in (
        "graph_trainer_deepseek_v4_debugmodel",
        "graph_trainer_deepseek_v4_flash",
        "graph_trainer_deepseek_v4_flash_43layers_16experts",
        "graph_trainer_deepseek_v4_pro",
        "graph_trainer_deepseek_v4_pro_61layers_32experts",
    ):
        assert "use_npu_config" not in _function_source(registry_path, graph_recipe)
