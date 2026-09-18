# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 trainer config recipes (standalone; no V4 imports).

Operator selection follows the DSV4 pattern through named recipes: the
``*_multimodal`` recipe builds the pure reference stack on real CC12M data, and
the ``*_multimodal_a3`` / ``*_multimodal_a5`` hardware recipes add the accepted fused
operators through ``override.imports`` — the printed Trainer Config is the
single source of truth for a run.  The default optimizer is AdamW and Muon
stays an explicit ``--optimizer.name`` selection.
"""

import dataclasses
from typing import TYPE_CHECKING, cast

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_npu.config import MuonOptimizerProfile, OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.profiler import CANNProfiler
from torchtitan_npu.extensions.trainer import TrainerEx
from torchtitan_npu.models.common.muon import make_expert_layout, make_owned_layout

from . import model_registry
from .dataloader import DeepSeekV41DataLoader

if TYPE_CHECKING:
    from .model import V41Model


def _document_alignment(model_spec) -> int:
    """Largest compression ratio the model pools with, ``1`` when it has none.

    A pooled group must never straddle a document edge: the row length stays a
    multiple of the pooling ratio, and the multimodal dataloader applies the
    same alignment to each packed document.
    """
    ratios = [ratio for ratio in model_spec.model.compress_ratios if ratio > 1]
    return max(ratios) if ratios else 1


def _v41_muon_profile(model_spec: ModelSpec) -> MuonOptimizerProfile:
    """Build the V4.1-owned parameter and FlexShard policy for Muon.

    Same sharding policy as the DSV4 profile where the parameter spaces
    overlap (per-head ``wq_b``/``wo_a`` BlockShard, expert-sharded routed
    GEMMs, Owned for the rest).  V4.1 differences: no MTP depths, no
    ``hc_head``, no compressor APE.  The indexer parameters do receive LI
    gradients on the new baseline, but they stay on the AdamW fallback group
    together with the vision tower and every 1-D parameter — this profile
    does not change the parameter policy the verified stacks ran with.
    """
    model_config = cast("V41Model.Config", model_spec.model)
    # FSDP folds the (dp_shard, cp) storage mesh into the single
    # ``dp_shard_cp`` axis when context parallelism is enabled, and the
    # partial_dtensor backend stores dense parameters on the plain ``fsdp``
    # axis (see parallelize.py).  Keep every axis name so the layouts match
    # whichever storage mesh the run actually uses.
    dense_dp_axes = (
        MeshAxisName.DP_SHARD.value,
        f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}",
        MeshAxisName.FSDP.value,
    )
    owned = make_owned_layout(dense_dp_axes)
    owned_attention_projections = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_projections = (*owned_attention_projections, "wq_b", "wo_a")
    expert_projections = ("w1", "w2", "w3")
    routed_expert_projections = ("w1_EFD", "w2_EDF", "w3_EFD")
    compressor_projections = ("wkv", "wgate")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    expert_sharding = make_expert_layout(dense_dp_axes)

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    bucket_configs: list[BucketConfig] = []
    for layer_id, layer_config in enumerate(model_config.layers):
        attention = layer_config.attention
        attention_shardings = {
            f"layers.{layer_id}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in owned_attention_projections.items()
        }
        # ``wq_b`` stores one [head_dim, q_lora_rank] matrix per head and
        # ``wo_a`` one [o_lora_rank, per_group_in] matrix per group, both
        # flattened on dim 0; BlockShard computes Muon per matrix.
        attention_shardings[f"layers.{layer_id}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={axis: BlockShard(dim=0, block_size=attention.head_dim) for axis in dense_dp_axes},
        )
        attention_shardings[f"layers.{layer_id}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_size=attention.wo_a.out_features) for axis in dense_dp_axes
            },
        )
        # Every layer carries a compressor; only the sources own weights.
        for projection in compressor_projections:
            if getattr(attention.compressor, projection, None) is not None:
                attention_shardings[f"layers.{layer_id}.attention.compressor.{projection}.weight"] = owned
        attention_shardings[f"layers.{layer_id}.hc_attn_pre.hc_fn"] = owned
        dense_shardings = {
            f"layers.{layer_id}.moe.shared_experts.{projection}.weight": owned for projection in expert_projections
        }
        dense_shardings[f"layers.{layer_id}.moe.router.gate.weight"] = owned
        dense_shardings[f"layers.{layer_id}.hc_ffn_pre.hc_fn"] = owned
        routed_shardings = {
            f"layers.{layer_id}.moe.routed_experts.inner_experts.{projection}": expert_sharding
            for projection in routed_expert_projections
        }
        # Split each layer into three buckets: the routed-expert GEMMs are
        # by far the largest and get a bucket of their own so the
        # Newton-Schulz all-gather peak stays bounded on the 40-layer
        # FSDP8 shape (a single per-layer bucket OOMs at ~89% HBM).
        compute_sharding_by_fqn.update(attention_shardings)
        compute_sharding_by_fqn.update(dense_shardings)
        compute_sharding_by_fqn.update(routed_shardings)
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.attn", patterns=tuple(attention_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.dense", patterns=tuple(dense_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.routed", patterns=tuple(routed_shardings)))

    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        rf"attention\.compressor\.(?:{'|'.join(compressor_projections)})\.weight|"
        rf"moe\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"moe\.routed_experts\.inner_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"moe\.router\.gate\.weight|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn"
        r")$"
    )
    return MuonOptimizerProfile(
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistMuon": {
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
                "bucket_configs": tuple(bucket_configs),
            }
        },
    )


def _v41_optimizer_config(model_spec: ModelSpec, *, lr: float) -> OptimizerConfig:
    """Build the AdamW-default V4.1 optimizer schema with a Muon profile.

    ``name`` stays ``AdamW`` unless the user explicitly selects
    ``--optimizer.name Muon``; the profile only arms that CLI selection and
    does not alter the default recipe.
    """
    adamw = default_adamw(lr=lr, eps=1e-6)
    return OptimizerConfig(
        lr=lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        muon_momentum=0.95,
        muon_enable_nesterov=True,
        muon_ns_steps=10,
        muon_adjust_lr_fn="match_rms_adamw",
        param_groups=adamw.param_groups,
        implementation=adamw.implementation,
        optimizer_factory_kwargs_by_name=adamw.optimizer_factory_kwargs_by_name,
        _muon_profile=_v41_muon_profile(model_spec),
    )


def _v41_trainer_config(
    flavor: str,
    *,
    seq_len: int = 512,
    fsdp_shard_degree: int = 8,
    context_parallel_degree: int = 1,
    expert_parallel_degree: int = 8,
    local_batch_size: int = 1,
    steps: int = 10,
    dataloader: ParallelAwareDataloader.Config,
    tokenizer: BaseTokenizer.Config,
) -> TrainerEx.Config:
    """Trainer recipe shared by the registered V4.1 single-node flavors.

    The flavors differ only in their model widths and layer count, which the
    model registry owns; the trainer shape below is the frozen V4.1
    single-node resource crop (FSDP 8 / EP 8, eager execution, FullAC).  The
    reference operators are the default; the ``*_multimodal_a3`` / ``*_multimodal_a5``
    recipes add the accepted fused operators through ``override.imports``.
    """
    model_spec = model_registry(flavor)
    # The registry validates the crop-level topology; this only pins that the
    # config the trainer is handed describes every layer it will build.
    if model_spec.model.n_layers != len(model_spec.model.layers):  # pyrefly: ignore [missing-attribute]
        raise ValueError("registered V4.1 model does not describe every configured layer")

    return TrainerEx.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=CANNProfiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        tokenizer=tokenizer,
        dataloader=dataloader,
        optimizer=_v41_optimizer_config(model_spec, lr=1e-5),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=fsdp_shard_degree,
            expert_parallel_degree=expert_parallel_degree,
            tensor_parallel_degree=1,
            context_parallel_degree=context_parallel_degree,
            pipeline_parallel_degree=1,
            fsdp_reshard_after_forward="always",
            context_parallel_load_balancer="headtail",
        ),
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(enable=False, interval=10),  # pyrefly: ignore [bad-argument-type]
    )


# The multimodal recipe family shares one run profile: run length and LR schedule
# defaults, parallelism runtime fields, deterministic seeding, comm timeouts
# and the virtual optimizer all live in _multimodal_trainer_config.  Short A/B
# runs override the schedule explicitly through the standard CLI
# (--training.steps with --lr-scheduler.total-steps and --warmup-steps).
_REFERENCE_IMPORTS = ("torchtitan_npu.override.common.optimizer.virtual",)
_A3_FUSED_IMPORTS = (
    "torchtitan_npu.override.common.rms_norm.asc",
    "torchtitan_npu.override.common.rope.asc_workaround",
    "torchtitan_npu.override.common.rope.asc_half_rotation",
    "torchtitan_npu.override.common.token_dispatcher.asc",
    "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post",
)
# A5-only kernels: the fused sparse attention (A3 rejects cmp_topk=256) and
# mHC Sinkhorn (operator package missing on A3 CANN 9.2.0).  The fused sparse
# port carries the layer's ratio and the metadata's document boundaries; the
# distillation loss stays owned by the model.
_A5_ONLY_IMPORTS = (
    "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
    "torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn",
)


def _multimodal_trainer_config() -> TrainerEx.Config:
    model_spec = model_registry("deepseek_v4_1_flash_40layers_16experts_vision")
    cfg = _v41_trainer_config(
        "deepseek_v4_1_flash_40layers_16experts_vision",
        dataloader=DeepSeekV41DataLoader.Config(
            document_alignment=_document_alignment(model_spec),
        ),
        tokenizer=HuggingFaceTokenizer.Config(),
    )
    cfg.training = dataclasses.replace(cfg.training, steps=40, global_batch_size=8)
    cfg.lr_scheduler = dataclasses.replace(cfg.lr_scheduler, warmup_steps=2, total_steps=40, decay_ratio=1.0)
    cfg.parallelism = dataclasses.replace(
        cfg.parallelism,
        spmd_backend="spmd_types",
        context_parallel_load_balancer=None,
        enable_sequence_parallel=True,
    )
    cfg.optimizer = dataclasses.replace(cfg.optimizer, implementation="fused")
    cfg.debug = dataclasses.replace(
        cfg.debug, seed=42, deterministic=True, print_config=True, moe_force_load_balance=False
    )
    cfg.comm = dataclasses.replace(cfg.comm, init_timeout_seconds=7200, train_timeout_seconds=600)
    cfg.override.imports += _REFERENCE_IMPORTS
    return cfg


def deepseek_v4_1_flash_40layers_16experts_multimodal() -> TrainerEx.Config:
    """40-layer multimodal recipe — the pure reference real-data entry.

    The WebDataset source comes in through ``--dataloader.dataset-path`` and
    the tokenizer through ``--hf-assets-path``; the trainer builds exactly
    one tokenizer and hands it to the dataloader.  The indexer distillation
    keeps the model's default coefficient (0.01).
    """
    return _multimodal_trainer_config()


def deepseek_v4_1_flash_40layers_16experts_multimodal_a3() -> TrainerEx.Config:
    """A3 hardware recipe: the reference entry plus the accepted fused
    stack (common RMSNorm, split-aware text rotary, vision half rotary,
    mHC post); the grouped MoE GEMMs are the common path reference runs
    share."""
    cfg = _multimodal_trainer_config()
    cfg.override.imports = [*_REFERENCE_IMPORTS, *_A3_FUSED_IMPORTS]
    return cfg


def deepseek_v4_1_flash_40layers_16experts_multimodal_a5() -> TrainerEx.Config:
    """A5 hardware recipe: the A3 stack plus the A5-only fused sparse
    attention and mHC Sinkhorn kernels."""
    cfg = deepseek_v4_1_flash_40layers_16experts_multimodal_a3()
    cfg.override.imports += _A5_ONLY_IMPORTS
    return cfg


def _debug_multimodal_trainer_config() -> TrainerEx.Config:
    """Debug-width multimodal entry for the committed upstream test tar and smokes."""
    model_spec = model_registry("deepseek_v4_1_debugmodel")
    cfg = _v41_trainer_config(
        "deepseek_v4_1_debugmodel",
        dataloader=DeepSeekV41DataLoader.Config(
            document_alignment=_document_alignment(model_spec),
        ),
        tokenizer=HuggingFaceTokenizer.Config(),
    )
    cfg.training = dataclasses.replace(cfg.training, steps=4, global_batch_size=2)
    cfg.lr_scheduler = dataclasses.replace(cfg.lr_scheduler, warmup_steps=2, total_steps=4)
    cfg.override.imports += _REFERENCE_IMPORTS
    return cfg


def deepseek_v4_1_debugmodel_multimodal() -> TrainerEx.Config:
    """Debug-width multimodal reference recipe (committed upstream test tar)."""
    return _debug_multimodal_trainer_config()


def deepseek_v4_1_debugmodel_multimodal_a3() -> TrainerEx.Config:
    """Debug-width A3 recipe: the test tar plus the accepted fused stack."""
    cfg = _debug_multimodal_trainer_config()
    cfg.override.imports = [*_REFERENCE_IMPORTS, *_A3_FUSED_IMPORTS]
    return cfg
