# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metadata extension and LI provider seams for model-built attention metadata.

The model owns its per-batch metadata construction (``build_attention_masks``
on the Decoder — including its own context-parallel handling).  The
``LightningIndexerMetadata`` provider owns ratio-4 LI metadata, while
``MetadataExtension`` owns the remaining backend-specific metadata.
"""

from dataclasses import dataclass, field

from torchtitan.config.configurable import Configurable


@dataclass(kw_only=True, slots=True)
class LightningIndexerKernelConfig:
    """DSV4 LI contract shared by metadata and kernel callers.

    The current DSV4 implementation supports only the ratio-4 TND contract.
    """

    layout_q: str = "TND"
    layout_k: str = "TND"
    mask_mode: int = 3
    cmp_ratio: int = 4

    def __post_init__(self) -> None:
        if (self.layout_q, self.layout_k, self.mask_mode, self.cmp_ratio) != ("TND", "TND", 3, 4):
            raise ValueError(
                "DeepSeek-V4 LightningIndexer supports only layout_q='TND', layout_k='TND', mask_mode=3, cmp_ratio=4"
            )


class MetadataExtension(Configurable):
    """The default identity post-process of the built attention masks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        window_size: int = 0
        """The sliding-window size (model-config constant), consumed by the
        vendor extension's kernel metadata fills."""
        num_heads: int | None = None
        head_dim: int | None = None
        index_n_heads: int | None = None
        index_head_dim: int | None = None
        index_topk: int | None = None
        """Static sparse-attention geometry consumed by vendor metadata
        extensions when the selected path requires it."""
        li_kernel_config: LightningIndexerKernelConfig = field(
            default_factory=LightningIndexerKernelConfig,
        )
        """Shared LI operator contract available to backend extensions."""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, metadata):
        return metadata


class LightningIndexerMetadata(Configurable):
    """Independent metadata provider for the LightningIndexer kernel.

    The provider is separate from ``MetadataExtension`` because BF16 and
    quantized LI kernels use different opaque metadata contracts. SMLA,
    SMLAG, and SLIG metadata remains owned by ``MetadataExtension``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        window_size: int = 0
        index_n_heads: int | None = None
        index_head_dim: int | None = None
        index_topk: int | None = None
        li_kernel_config: LightningIndexerKernelConfig = field(
            default_factory=LightningIndexerKernelConfig,
        )
        """LI layout, mask, and compression-ratio contract for the provider."""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, metadata):
        return metadata
