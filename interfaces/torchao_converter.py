# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-NPU converter integration hosted by ``torchtitan-npu``.

TorchTitan builds models from a tree of ``Module.Config`` objects.  Quantization
must therefore be expressed in that tree before the model is built: a converter
replaces selected configs with configs whose modules install torchao-npu
parameter wrappers or module-level QAT behavior in ``__init__``.  This keeps
the transforms in place before
TP/EP/FSDP and optimizer construction without modifying TorchTitan's trainer.

Only this adapter lives in ``torchtitan_npu``.  Quantization configs, parameter
wrappers and NPU kernels continue to come from the separately installed
``torchao_npu`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Protocol, cast

import torch
import tyro
from torchao.core.config import AOBaseConfig  # noqa: TC002
from torchao.quantization.quant_api import quantize_
from torchao_npu.configs import ParamSwapConfig
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    FP8QuantizeConfig,
    HiF8QuantizeConfig,
    MXQuantizeConfig,
)
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.config import derive
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.utils import validate_converter_order
from torchtitan.tools.logging import logger

from torchtitan_npu.config.configs import LIQuantization  # noqa: TC001 - Tyro resolves the converter Config at runtime.
from torchtitan_npu.models.common.metadata_extension import (
    LightningIndexerKernelConfig,
    LightningIndexerMetadata,
)
from torchtitan_npu.models.deepseek_v4.compressor import LightningIndexer
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

if TYPE_CHECKING:
    from torchao_npu.configs import QuantLightningIndexerConfig
    from torchtitan.protocols.model_spec import ModelSpec
    from torchtitan.protocols.module import Module

    from torchtitan_npu.config.configs import QuantizationExtensionConfig


class ConfigFilterFn(Protocol):
    """Predicate over a TorchTitan model-config node and its config-tree FQN."""

    def __call__(self, config: Module.Config, fqn: str) -> bool: ...


def any_config_filter(*filters: ConfigFilterFn) -> ConfigFilterFn:
    """Combine config-tree filters with logical OR."""

    def _filter(config: Module.Config, fqn: str) -> bool:
        return any(filter_fn(config, fqn) for filter_fn in filters)

    return _filter


def match_config_fqn_suffix(*suffixes: str) -> ConfigFilterFn:
    """Match config-tree FQNs by a dotted suffix or an exact root name.

    ``Config.traverse`` reports nested fields as dotted FQNs, but a field on
    the model config itself has no leading component (for example,
    ``lightning_indexer_metadata``).  Accepting both forms keeps filters
    useful for root-owned extension configs as well as nested modules.
    """

    normalized = tuple(suffix.lstrip(".") for suffix in suffixes)

    def _filter(config: Module.Config, fqn: str) -> bool:
        return any(fqn == suffix or fqn.endswith(f".{suffix}") for suffix in normalized)

    return _filter


def _replace_config(model_config, parent: object | None, attr: str | int | None, replacement):
    """Replace one config-tree node, including the uncommon root-node case."""

    if parent is None:
        return replacement
    if isinstance(parent, list):
        assert isinstance(attr, int)
        parent[attr] = replacement
    else:
        assert isinstance(attr, str)
        setattr(parent, attr, replacement)
    return model_config


_npu_quantized_module_cache: dict[type[Module], type[Module]] = {}
_DEFAULT_TARGET_CONFIG_TYPES = (
    Linear.Config,
    BatchedLinear.Config,
    GroupedExperts.Config,
    LightningIndexer.Config,
    LightningIndexerMetadata.Config,
)


def _get_npu_quantized_module_cls(parent_cls: type[Module]) -> type[Module]:
    """Create a parameter-quantized subclass while preserving host behavior.

    Both Linear and GroupedExperts can have model-specific subclasses.  The
    generated class inherits the concrete config owner instead of replacing it
    with a fixed implementation, so custom forward methods and config fields
    (for example DeepSeek-V4's SwiGLU clamp) remain intact.
    """

    if parent_cls in _npu_quantized_module_cache:
        return _npu_quantized_module_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    class NpuQuantizedModule(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc, valid-type]
            # ParamSwapConfig contains Python objects and is supplied by the
            # recipe, not by Tyro CLI parsing or config serialization.
            _torchao_npu_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            if config._torchao_npu_config is None:
                raise ValueError(f"{type(self).__name__}.Config requires _torchao_npu_config")
            quantize_(
                self,
                config._torchao_npu_config,
                filter_fn=lambda candidate, _fqn: candidate is self,
            )

    NpuQuantizedModule.__name__ = f"NpuQuantized{parent_cls.__name__}"
    NpuQuantizedModule.__qualname__ = f"NpuQuantized{parent_cls.__name__}"
    _npu_quantized_module_cache[parent_cls] = NpuQuantizedModule
    return NpuQuantizedModule


class NpuQuantizeConverter(QuantizationConverter):
    """Replace selected quantizable module or metadata config nodes.

    A recipe may instantiate this converter more than once with different
    parameter-swap policies.  For example, one instance can select attention
    and shared-expert Linear nodes for MX training while another selects routed
    GroupedExperts nodes for Block FP8 training.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        base_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None
        filter_fn: Annotated[ConfigFilterFn | None, tyro.conf.Suppress] = None
        replacement_config_type: Annotated[type | None, tyro.conf.Suppress] = None
        replacement_kwargs: Annotated[dict[str, object], tyro.conf.Suppress] = field(default_factory=dict)
        require_match: bool = True

        def __post_init__(self) -> None:
            if self.base_config is None and self.replacement_config_type is None:
                raise ValueError("NpuQuantizeConverter.Config requires base_config or replacement_config_type")

    def __init__(self, config: Config) -> None:
        self.config = config

    def convert(self, model_config):
        converted = 0
        matches = []
        seen_fqns = set()
        for config_type in _DEFAULT_TARGET_CONFIG_TYPES:
            for match in model_config.traverse(config_type):
                if match[0] not in seen_fqns:
                    matches.append(match)
                    seen_fqns.add(match[0])
        for fqn, config, parent, attr in matches:
            if self.config.filter_fn is not None and not self.config.filter_fn(config, fqn):
                continue
            if self.config.replacement_config_type is not None:
                replacement = derive(config, self.config.replacement_config_type, **self.config.replacement_kwargs)
                replacement_name = self.config.replacement_config_type.__qualname__
            else:
                # Metadata is a Configurable provider, not an nn.Module.  It
                # shares the target set only so the replacement branch can
                # handle it; parameter quantization must skip it.
                if isinstance(config, LightningIndexerMetadata.Config):
                    continue
                assert self.config.base_config is not None
                parent_cls = cast("type[Module] | None", type(config)._owner)
                if parent_cls is None:
                    raise TypeError(f"Config at {fqn!r} has no owning module class")
                if parent_cls in _npu_quantized_module_cache.values():
                    continue

                quantized_cls = _get_npu_quantized_module_cls(parent_cls)
                replacement = derive(
                    config,
                    quantized_cls.Config,
                    _torchao_npu_config=self.config.base_config,
                )
                replacement_name = f"{quantized_cls.__qualname__}.Config"
            model_config = _replace_config(model_config, parent, attr, replacement)
            converted += 1
            logger.info(
                "[Converter] %s.%s: model_spec.model.%s %s -> %s",
                type(self).__module__,
                type(self).__qualname__,
                fqn,
                type(config).__qualname__,
                replacement_name,
            )

        if converted == 0 and self.config.require_match:
            raise ValueError("NpuQuantizeConverter did not match any config nodes")
        logger.info("Converted %d config node(s) for torchao-npu", converted)
        return model_config


# DeepSeek-V4 config-tree filters for the current model hierarchy.
_DSV4_CONFIG_FILTERS = {
    "dense": match_config_fqn_suffix(
        ".attention.wq_a",
        ".attention.wq_b",
        ".attention.wkv",
        ".attention.wo_a",
        ".attention.wo_b",
        ".attention.indexer.wq_b",
        ".moe.shared_experts.w1",
        ".moe.shared_experts.w2",
        ".moe.shared_experts.w3",
    ),
    "routed_expert": match_config_fqn_suffix(".moe.routed_experts.inner_experts"),
    "lightning_indexer": match_config_fqn_suffix(".attention.compressed_sparse_attention.lightning_indexer"),
    "lightning_indexer_metadata": match_config_fqn_suffix(".lightning_indexer_metadata"),
}


_SUPPORTED_RECIPES = ("all_mxfp8", "mix", "all_block_fp8")


def _prepare_quant_lightning_indexer_inputs(
    module,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    attention_masks,
):
    from torchao_npu.quantized_modules.lightning_indexer import QuantizedLightningIndexerInputs

    ratio = module._torchao_npu_module_swap_config.cmp_ratio
    plan = attention_masks.plans.get(ratio)
    if plan is None:
        raise ValueError(f"LI compression ratio {ratio} is not present in attention metadata")
    if plan.li_metadata is None:
        raise RuntimeError("Quantized LI requires the quantized LI metadata provider")
    key = idx_k.flatten(0, 1)
    if plan.cmp_k_global_gather_indices is not None:
        key = key[plan.cmp_k_global_gather_indices]
    else:
        key = key[: plan.n_cmp_blocks_host]
    return QuantizedLightningIndexerInputs(
        query=idx_q.flatten(0, 1),
        key=key.unsqueeze(1).contiguous(),
        weights=idx_w.flatten(0, 1).float(),
        topk=module.index_topk,
        metadata=plan.li_metadata,
        cu_seqlens_q=attention_masks.varlen.cu_seq_q,
        cu_seqlens_k=plan.cu_seqlens_cmp_k,
        cmp_residual_k=plan.block_remainder,
    )


class _QuantizedLightningIndexerMetadataAdapter(LightningIndexerMetadata):
    """Bridge TorchTitan metadata to the TorchAO-NPU LI provider."""

    @dataclass(kw_only=True, slots=True)
    class Config(LightningIndexerMetadata.Config):
        index_n_heads: int  # pyrefly: ignore [bad-override]
        index_head_dim: int  # pyrefly: ignore [bad-override]
        index_topk: int  # pyrefly: ignore [bad-override]
        quant_mode: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        from torchao_npu.quantized_modules.lightning_indexer_metadata import (
            QuantizedLightningIndexerMetadata,
            QuantizedLightningIndexerMetadataConfig,
        )

        cfg = cast("_QuantizedLightningIndexerMetadataAdapter.Config", config)
        kernel = cfg.li_kernel_config
        self._provider = QuantizedLightningIndexerMetadata(
            QuantizedLightningIndexerMetadataConfig(
                index_n_heads=cfg.index_n_heads,
                index_head_dim=cfg.index_head_dim,
                index_topk=cfg.index_topk,
                layout_q=kernel.layout_q,
                layout_k=kernel.layout_k,
                mask_mode=kernel.mask_mode,
                cmp_ratio=kernel.cmp_ratio,
                quant_mode=cfg.quant_mode,
            )
        )

    def __call__(self, metadata):
        ratio = self.config.li_kernel_config.cmp_ratio
        plan = metadata.plans.get(ratio)
        if plan is None:
            raise ValueError(f"LI compression ratio {ratio} is not present in attention metadata")
        if plan.gather_indices.numel() == 0:
            raise ValueError("batch has no complete compression block for the configured LI ratio")
        plan.li_metadata = self._provider(
            cu_seqlens_q=metadata.varlen.cu_seq_q,
            cu_seqlens_k=plan.cu_seqlens_cmp_k,
            cmp_residual_k=plan.block_remainder,
        )
        return metadata


def _quant_lightning_indexer(
    kernel_config: LightningIndexerKernelConfig,
    *,
    li_quantization: LIQuantization,
    dst_type_max: float = 0.0,
) -> QuantLightningIndexerConfig:
    from torchao_npu.configs import QuantLightningIndexerConfig, quant_mode_for_config

    qk_config: FP8QuantizeConfig | HiF8QuantizeConfig | MXQuantizeConfig
    if li_quantization == "fp8":
        qk_config = FP8QuantizeConfig()
    elif li_quantization == "mxfp4":
        qk_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    elif li_quantization == "mxfp8":
        qk_config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    elif li_quantization == "hif8":
        qk_config = HiF8QuantizeConfig(dst_type_max=dst_type_max)
    else:
        raise ValueError("Quantized LI requires mxfp4, mxfp8, fp8, or hif8")
    quant_mode = quant_mode_for_config(qk_config)
    return QuantLightningIndexerConfig(
        layout_q=kernel_config.layout_q,
        layout_k=kernel_config.layout_k,
        mask_mode=kernel_config.mask_mode,
        cmp_ratio=kernel_config.cmp_ratio,
        query_config=qk_config,
        key_config=qk_config,
        quant_mode=quant_mode,
        input_adapter=_prepare_quant_lightning_indexer_inputs,
    )


def _mxfp8_param_swap() -> ParamSwapConfig:
    mx_config = MXQuantizeConfig()
    return ParamSwapConfig(
        weight_config=mx_config,
        activation_config=mx_config,
    )


def _block_fp8_param_swap(
    *,
    enable_mxfp4_qat: bool = False,
    dst_type_max: float = 0.0,
    fsdp_prequantize: bool = False,
) -> ParamSwapConfig:
    mxfp4_config = None
    if enable_mxfp4_qat:
        mxfp4_config = MXQuantizeConfig(
            elem_dtype=torch.float4_e2m1fn_x2,
            dst_type_max=dst_type_max,
        )
    return ParamSwapConfig(
        weight_config=BlockMXQuantizeConfig(
            mxfp4_fake_quantize_config=mxfp4_config,
            fsdp_prequantize=fsdp_prequantize,
        ),
        activation_config=MXQuantizeConfig(),
    )


def _quantization_converter(
    base_config: AOBaseConfig | None,
    filter_fn: ConfigFilterFn | None = None,
    *,
    model_compile_enabled: bool,
    replacement_config_type: type | None = None,
    replacement_kwargs: dict[str, object] | None = None,
) -> NpuQuantizeConverter.Config:
    return NpuQuantizeConverter.Config(
        base_config=base_config,
        filter_fn=filter_fn,
        model_compile_enabled=model_compile_enabled,
        replacement_config_type=replacement_config_type,
        replacement_kwargs=replacement_kwargs or {},
    )


def _recipe_converters(
    recipe: str,
    *,
    enable_mxfp4_qat: bool,
    dst_type_max: float,
    fsdp_prequantize: bool,
    model_compile_enabled: bool,
    li_quantization: LIQuantization | None = None,
    li_kernel_config: LightningIndexerKernelConfig | None = None,
) -> list[QuantizationConverter.Config]:
    converters: list[QuantizationConverter.Config]
    dense_filter = _DSV4_CONFIG_FILTERS["dense"]

    if recipe == "all_mxfp8":
        converters = [
            _quantization_converter(
                _mxfp8_param_swap(),
                any_config_filter(dense_filter, _DSV4_CONFIG_FILTERS["routed_expert"]),
                model_compile_enabled=model_compile_enabled,
            )
        ]

    else:
        routed_config = _block_fp8_param_swap(
            enable_mxfp4_qat=enable_mxfp4_qat,
            dst_type_max=dst_type_max,
            fsdp_prequantize=fsdp_prequantize,
        )
        if recipe == "mix":
            dense_config = _mxfp8_param_swap()
        elif recipe == "all_block_fp8":
            dense_config = _block_fp8_param_swap(fsdp_prequantize=fsdp_prequantize)
        else:
            raise ValueError(f"recipe must be one of {_SUPPORTED_RECIPES}, got {recipe!r}")

        converters = [
            _quantization_converter(
                dense_config,
                dense_filter,
                model_compile_enabled=model_compile_enabled,
            ),
            _quantization_converter(
                routed_config,
                _DSV4_CONFIG_FILTERS["routed_expert"],
                model_compile_enabled=model_compile_enabled,
            ),
        ]

    if li_quantization is not None:
        li_config = _quant_lightning_indexer(
            li_kernel_config or LightningIndexerKernelConfig(),
            li_quantization=li_quantization,
            dst_type_max=dst_type_max,
        )
        converters.append(
            _quantization_converter(
                li_config,
                _DSV4_CONFIG_FILTERS["lightning_indexer"],
                model_compile_enabled=model_compile_enabled,
            )
        )
        converters.append(
            _quantization_converter(
                None,
                _DSV4_CONFIG_FILTERS["lightning_indexer_metadata"],
                model_compile_enabled=model_compile_enabled,
                replacement_config_type=_QuantizedLightningIndexerMetadataAdapter.Config,
                replacement_kwargs={"quant_mode": li_config.quant_mode},
            )
        )
    return converters


def _get_li_kernel_config(model_config) -> LightningIndexerKernelConfig:
    metadata_matches = [
        match
        for match in model_config.traverse(LightningIndexerMetadata.Config)
        if _DSV4_CONFIG_FILTERS["lightning_indexer_metadata"](match[1], match[0])
    ]
    li_matches = [
        match
        for match in model_config.traverse(LightningIndexer.Config)
        if _DSV4_CONFIG_FILTERS["lightning_indexer"](match[1], match[0])
    ]
    if len(metadata_matches) != 1:
        raise ValueError("Quantized LI requires one lightning_indexer_metadata config")
    if not li_matches:
        raise ValueError("Quantized LI requires a DeepSeek-V4 LightningIndexer config")

    metadata = metadata_matches[0][1]
    for fqn, li_config, _, _ in li_matches:
        if li_config.li_kernel_config != metadata.li_kernel_config or li_config.index_topk != metadata.index_topk:
            raise ValueError(f"LightningIndexer at {fqn!r} disagrees with its metadata config")
    return metadata.li_kernel_config


def apply_quantization_converter(
    model_spec: ModelSpec | None,
    quantization_config: QuantizationExtensionConfig,
    *,
    model_compile_enabled: bool,
) -> ModelSpec:
    """Apply the selected TorchAO-NPU recipe to an existing BF16 model spec.

    This runs during ``TrainerEx`` construction and before base ``Trainer``
    initialization.  The regular model registry therefore remains responsible
    only for constructing the high-precision config tree, while this function
    performs the optional low-precision config replacement.
    """

    if model_spec is None:
        raise ValueError("TorchAO-NPU quantization requires model_spec to be configured")
    if not quantization_config.enable_quantized_training:
        return model_spec
    quantization_config.validate()

    li_kernel_config = (
        _get_li_kernel_config(model_spec.model) if quantization_config.li_quantization is not None else None
    )
    converters = _recipe_converters(
        quantization_config.recipe,
        enable_mxfp4_qat=quantization_config.enable_mxfp4_qat,
        dst_type_max=quantization_config.dst_type_max,
        fsdp_prequantize=quantization_config.fsdp_prequantize,
        model_compile_enabled=model_compile_enabled,
        li_quantization=quantization_config.li_quantization,
        li_kernel_config=li_kernel_config,
    )
    validate_converter_order(converters)

    model_config = model_spec.model
    for converter_config in converters:
        model_config = converter_config.build().convert(model_config)

    logger.info(
        "Applied TorchAO-NPU recipe=%s, mxfp4_qat=%s, li_quantization=%s, dst_type_max=%s, fsdp_prequantize=%s",
        quantization_config.recipe,
        quantization_config.enable_mxfp4_qat,
        quantization_config.li_quantization,
        quantization_config.dst_type_max,
        quantization_config.fsdp_prequantize,
    )
    return replace(model_spec, model=model_config)


__all__ = [
    "ConfigFilterFn",
    "NpuQuantizeConverter",
    "any_config_filter",
    "apply_quantization_converter",
    "match_config_fqn_suffix",
]
