# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CANN compressor using the operator's native autograd implementation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Any

import torch

from torchtitan_npu.models.deepseek_v4.compressor import Compressor, CompressorImplementation
from torchtitan_npu.models.deepseek_v4.metadata import CompressedVarlenMetadata


def _state_inputs(cu_seqlens: torch.Tensor, x: torch.Tensor, *, max_length: int, ratio: int, head_dim: int):
    cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous()
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    block_size = 8 if ratio == 4 else 16
    coff = 2 if ratio == 4 else 1
    # CANN tiling requires state_block_table despite its optional schema.
    # In cache_mode=1, block ID 0 skips cache writes for full-document training.
    state_block_table = torch.zeros(
        (lengths.numel(), max((max_length + block_size - 1) // block_size, 1)),
        dtype=torch.int32,
        device=x.device,
    )
    state_cache = torch.zeros((1, block_size, 2 * coff * head_dim), dtype=torch.float32, device=x.device)
    return state_block_table, state_cache, cu_seqlens, lengths


class AscCompressor(CompressorImplementation):
    @dataclass(kw_only=True, slots=True)
    class Config(CompressorImplementation.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Import registers the operator when this CANN package provides one.
        compressor_module = importlib.import_module("cann_ops_transformer.ops.compressor")
        compressor_op = getattr(torch.ops.cann_ops_transformer, "compressor", None)
        self._compressor_fn = compressor_op.default if compressor_op is not None else compressor_module.compressor

    def forward(self, compressor: Compressor, x: torch.Tensor, attention_masks: Any) -> torch.Tensor:
        plan = attention_masks.plans[compressor.compress_ratio]
        if plan.gather_indices is None or plan.block_positions is None:
            raise ValueError("CANN compressor requires complete compression metadata")
        if plan.exchange is not None:
            # Exchange only missing boundary tokens before the fused projections.
            # Plan segments include the C4 predecessor block; restarting each
            # segment masks its first overlap exactly as the reference does.
            max_length = min(plan.gather_indices.numel(), x.shape[1] + 2 * compressor.compress_ratio)
            x = compressor.token_dispatcher.gather(x, plan)
            if plan.gather_indices.numel() == 0:
                # CompressorGrad rejects empty inputs. The collective must still
                # run, including on ranks with no local compressed blocks.
                local_metadata = CompressedVarlenMetadata(
                    varlen=attention_masks.varlen,
                    plans={compressor.compress_ratio: replace(plan, exchange=None)},
                )
                return compressor._forward(x, local_metadata)
            assert plan.first_indices is not None
            cu_seqlens = torch.cat(
                (
                    plan.first_indices * compressor.compress_ratio,
                    plan.first_indices.new_full((1,), plan.gather_indices.numel()),
                )
            )
        else:
            if plan.gather_indices.numel() == 0:
                return compressor._forward(x, attention_masks)
            cu_seqlens = attention_masks.varlen.cu_seq_q
            max_length = attention_masks.varlen.max_k

        state_block_table, state_cache, cu_seqlens, lengths = _state_inputs(
            cu_seqlens, x, max_length=max_length, ratio=compressor.compress_ratio, head_dim=compressor.head_dim
        )
        pooled = self._compressor_fn(
            x.reshape(-1, x.shape[-1]).contiguous(),
            compressor.wkv.weight,
            compressor.wgate.weight,
            state_cache,
            compressor.ape,
            cmp_ratio=compressor.compress_ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=lengths,
            coff=1 + int(compressor.overlap),
            cache_mode=1,
        )
        pooled = compressor.norm(pooled[: plan.gather_indices.numel() // compressor.compress_ratio].to(x.dtype))
        return (
            compressor.rope(pooled.unsqueeze(0).unsqueeze(2), positions=plan.block_positions.unsqueeze(0))
            .squeeze(0)
            .squeeze(1)
        )
