# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/5769
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/6412

"""Use signed 64-bit SDC replay digests and support sparse COO tensors.

This module is the sole patch owner of ``SDCReplayer._signature``; the upstream
constructor, capability probe and replay lifecycle remain unchanged. Remove this
module after the TorchTitan dependency includes the PRs.
"""

from __future__ import annotations

import random

import torch
import torchtitan.observability.sdc_replayer
from torchtitan.observability.sdc_replayer import (
    SDCReplayer,
    _accelerator_rng_states,
    _local_tensor,
    _ReplaySignature,
)

_FNV64_PRIME = 0x100000001B3


def _npu_signature(self: SDCReplayer, loss: torch.Tensor) -> _ReplaySignature:
    # pylint: disable=protected-access
    schema: list[tuple[str, tuple[int, ...], str, str]] = []
    digests: list[torch.Tensor] = []
    digest_device = _local_tensor(loss).device

    def add_tensor(name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            schema.append((f"{name}:none", (), "none", "none"))
            digests.append(torch.zeros((), dtype=torch.int64, device=digest_device))
            return
        local = _local_tensor(tensor.detach())
        schema.append((name, tuple(local.shape), str(local.dtype), str(local.device)))
        if local.numel() == 0:
            digests.append(torch.zeros((), dtype=torch.int64, device=digest_device))
            return
        if local.layout == torch.sparse_coo:
            coalesced = local.coalesce()
            # Separate torch.hash_tensor calls on indices and values use order-insensitive
            # XOR and miss value swaps between rows. Weight stored words in coalesced order
            # by powers of the 64-bit FNV prime modulo 2**64 (upstream #6412).
            word_dtype = {1: torch.uint8, 2: torch.int16}.get(coalesced.values().element_size(), torch.int32)
            words = torch.cat(
                [
                    part.clone(memory_format=torch.contiguous_format).view(-1).view(word_dtype)
                    for part in (coalesced.indices(), coalesced.values())
                ]
            )
            weights = torch.full_like(words, _FNV64_PRIME, dtype=torch.int64).cumprod(0)
            # Include the index count to distinguish identical words split at a
            # different boundary between coordinates and values.
            digest = weights.mul_(words).sum() + words.numel() + coalesced.indices().numel()
            digests.append(digest.to(digest_device))
            return
        if local.is_complex():
            local = torch.view_as_real(local)
        # Preserve all 64 hash bits and use the signed representation for
        # NPU transfer/comparison, including hashes of CPU RNG state.
        digests.append(torch.hash_tensor(local).view(torch.int64).to(digest_device))

    add_tensor("loss", loss)
    add_tensor("rng:cpu", torch.get_rng_state())
    accelerator_rng_states = _accelerator_rng_states(self._device)
    if accelerator_rng_states is not None:
        for index, rng_state in enumerate(accelerator_rng_states):
            add_tensor(f"rng:accelerator:{index}", rng_state)

    seen_buffers: set[tuple[int, str]] = set()
    for module_index, root in enumerate(self._modules):
        for module_name, module in root.named_modules():
            for buffer_name, buffer in module._buffers.items():
                buffer_key = (id(module), buffer_name)
                if buffer_key in seen_buffers:
                    continue
                seen_buffers.add(buffer_key)
                qualified_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                add_tensor(f"buffer:{module_index}:{qualified_name}", buffer)

    seen_parameters: set[int] = set()
    for module_index, module in enumerate(self._modules):
        for name, parameter in module.named_parameters():
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            add_tensor(f"gradient:{module_index}:{name}", parameter.grad)

    state = (("state:python_rng", random.getstate()),)
    return _ReplaySignature(tuple(schema), tuple(digests), state)


def apply() -> None:
    torchtitan.observability.sdc_replayer.SDCReplayer._signature = _npu_signature  # pylint: disable=protected-access


apply()
