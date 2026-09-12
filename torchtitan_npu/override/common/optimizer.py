# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overrides for NPU swap-backed optimizer states and their checkpoints."""

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch_npu
from torch.distributed._tensor import DTensor
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import derive, override
from torchtitan.distributed.flex_shard.dist_muon import DistMuon

from torchtitan_npu.extensions.novaswap import swap_api

_ADAMW_SWAP_BUCKET_TIMES = 16


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _replace_local_tensor(tensor: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, DTensor):
        return local
    return DTensor.from_local(
        local,
        tensor.device_mesh,
        tensor.placements,
        shape=tensor.size(),
        stride=tensor.stride(),
        run_check=False,
    )


def make_swap_state_name(*parts: object) -> str:
    return ".".join(map(str, parts))


@dataclass(frozen=True)
class _AdamWSwapBucket:
    group: dict[str, Any]
    parameters: tuple[torch.Tensor, ...]
    state_name: str | None


class _NovaSwapAdamW:
    """Run stock AdamW state through NovaSwap with a bucket-level pipeline."""

    _state_keys = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    def __init__(self, optimizer: torch.optim.AdamW) -> None:
        self.optimizer = optimizer
        self._original_step = optimizer.step
        self._buckets: tuple[_AdamWSwapBucket, ...] = ()

    @staticmethod
    def _submit(bucket: _AdamWSwapBucket, action: str) -> None:
        if bucket.state_name is not None:
            swap_api.execute(bucket.state_name, action)

    def step(self, closure=None):
        if not self._buckets:
            loss = self._original_step(closure)
            self._buckets = self._build_buckets()
            for bucket in self._buckets:
                self._submit(bucket, "D2H")
            return loss

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._submit(self._buckets[0], "H2D")
        for bucket_index, bucket in enumerate(self._buckets):
            self._submit(bucket, "WAIT_DEVICE")
            if bucket_index + 1 < len(self._buckets):
                self._submit(self._buckets[bucket_index + 1], "H2D")
            self._update_bucket(bucket)
            self._submit(bucket, "D2H")
        return loss

    def _build_buckets(self) -> tuple[_AdamWSwapBucket, ...]:
        parameters: list[torch.Tensor] = []
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if "exp_avg" in self.optimizer.state[parameter]:
                    parameters.append(parameter)
        total_numel = sum(_local_tensor(parameter).numel() for parameter in parameters)
        bucket_numel_limit = max(total_numel // _ADAMW_SWAP_BUCKET_TIMES, 1)
        buckets: list[_AdamWSwapBucket] = []
        current_group: dict[str, Any] | None = None
        current_parameters: list[torch.Tensor] = []
        current_numel = 0
        current_signature: tuple[torch.device, torch.dtype] | None = None

        def append_bucket() -> None:
            if not current_parameters:
                return
            if current_group is None:
                raise RuntimeError("AdamW swap bucket has no parameter group")
            moments: list[tuple[dict[str, Any], str, torch.Tensor, torch.Tensor]] = []
            for parameter in current_parameters:
                state = self.optimizer.state[parameter]
                for state_key in self._state_keys:
                    moment = state.get(state_key)
                    if moment is None:
                        continue
                    local_moment = _local_tensor(moment)
                    if local_moment.numel() == 0:
                        continue
                    moments.append((state, state_key, moment, local_moment))

            state_name = None
            if moments:
                device, dtype = moments[0][3].device, moments[0][3].dtype
                if not all(
                    local_moment.device == device and local_moment.dtype == dtype for _, _, _, local_moment in moments
                ):
                    raise RuntimeError("AdamW swap bucket states must share a device and dtype")
                flat = torch.empty(
                    sum(local_moment.numel() for _, _, _, local_moment in moments),
                    dtype=dtype,
                    device=device,
                )
                offset = 0
                for state, state_key, moment, local_moment in moments:
                    flat_view = flat.narrow(0, offset, local_moment.numel()).view_as(local_moment)
                    flat_view.copy_(local_moment)
                    state[state_key] = _replace_local_tensor(moment, flat_view)
                    offset += local_moment.numel()
                state_name = make_swap_state_name("adamw", id(self.optimizer), "bucket", len(buckets))
                swap_api.register_tensor(flat, state_name)
            buckets.append(
                _AdamWSwapBucket(
                    group=current_group,
                    parameters=tuple(current_parameters),
                    state_name=state_name,
                )
            )

        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if "exp_avg" not in self.optimizer.state[parameter]:
                    continue
                parameter_numel = _local_tensor(parameter).numel()
                local_exp_avg = _local_tensor(self.optimizer.state[parameter]["exp_avg"])
                parameter_signature = (local_exp_avg.device, local_exp_avg.dtype)
                if current_parameters:
                    same_group = group is current_group
                    fits_bucket = current_numel + parameter_numel <= bucket_numel_limit
                    same_signature = parameter_signature == current_signature
                    if not (same_group and fits_bucket and same_signature):
                        append_bucket()
                        current_parameters = []
                        current_numel = 0
                current_group = group
                current_signature = parameter_signature
                current_parameters.append(parameter)
                current_numel += parameter_numel
        append_bucket()
        return tuple(buckets)

    def _update_bucket(self, bucket: _AdamWSwapBucket) -> None:
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [{**bucket.group, "params": bucket.parameters}]
        try:
            # The saved method is the unwrapped upstream AdamW.step.  Calling
            # it with exactly one temporary param group preserves all current
            # AdamW behavior while leaving this wrapper responsible only for
            # bucket scheduling.
            self._original_step()
        finally:
            self.optimizer.param_groups = original_param_groups


def _make_swap(t: torch.Tensor) -> torch.Tensor:
    local = t.to_local() if isinstance(t, DTensor) else t
    # A DTensor may have p.numel() > 0 globally but local.numel() == 0 on ranks
    # that own an empty shard. Swap-memory allocation rejects zero-sized tensors,
    # so preserve such local shards with a regular empty tensor instead.
    out = (
        torch.empty_like(local)
        if local.numel() == 0
        else torch_npu.empty_with_swapped_memory(local.size(), dtype=local.dtype, device=local.device)
    )
    return _replace_local_tensor(t, out)


def _swap_state_init_hook(optimizer, args, kwargs):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=p.device,
                )
                state["exp_avg"] = _make_swap(p).zero_()
                state["exp_avg_sq"] = _make_swap(p).zero_()


class VirtualOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts):
        super().__init__(config=config, model_parts=model_parts)
        for opt in self.optimizers:
            opt.register_step_pre_hook(_swap_state_init_hook)


@override(
    target=OptimizersContainer.Config,
    description="Allocate Adam/AdamW states in swap memory (host-offload)",
)
def virtual(
    cfg: OptimizersContainer.Config,
) -> VirtualOptimizersContainer.Config:
    return derive(cfg, VirtualOptimizersContainer.Config)


class OptimizerStateSwapContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts: list[Any]) -> None:
        super().__init__(config=config, model_parts=model_parts)
        parameter_to_part = {
            id(parameter): part_index
            for part_index, model_part in enumerate(model_parts)
            for parameter in model_part.parameters()
        }
        for optimizer in self.optimizers:
            if isinstance(optimizer, DistMuon):
                part_indices = {
                    parameter_to_part[id(parameter)]
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
                if len(part_indices) != 1:
                    raise RuntimeError(
                        "DistMuon optimizer must own parameters from exactly "
                        f"one model part, got part indices {sorted(part_indices)}"
                    )
                self._swap_muon(optimizer, next(iter(part_indices)))
            elif isinstance(optimizer, torch.optim.AdamW):
                self._swap_adamw(optimizer)

    @staticmethod
    def _swap_muon(optimizer: DistMuon, model_part: int) -> None:
        original_momentum = optimizer._momentum
        original_prepare_local = optimizer._prepare_local
        runtime = optimizer._redistribution_runtime
        original_enqueue_storage_to_compute = runtime._enqueue_storage_to_compute
        deferred_d2h_names: list[str] | None = None

        def momentum(optimizer_self, compute_layout, grad):
            state = optimizer_self.state[compute_layout.param]
            created = "momentum_buffer" not in state
            result = original_momentum(compute_layout, grad)
            if not created:
                return result
            local = result.to_local()
            if local.numel() == 0:
                return result
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            swap_api.register_tensor(local, name)
            swap_api.execute(name, "D2H")
            return result

        def submit_h2d(compute_layout) -> None:
            state = optimizer.state.get(compute_layout.param)
            if state is None or "momentum_buffer" not in state:
                return
            momentum = state["momentum_buffer"]
            local = momentum.to_local() if isinstance(momentum, DTensor) else momentum
            if local.numel() == 0:
                return
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            if swap_api.get_handle_phase(name) != "H2D":
                swap_api.execute(name, "H2D")

        def prefetch_plan(plan) -> None:
            """Submit a plan's H2D work before FlexShard enters its transfer stream."""
            redistributed = getattr(plan, "redistributed_items", None)
            items = plan.items if redistributed is None else (*redistributed, *plan.unredistributed_items)
            for compute_layout in items:
                submit_h2d(compute_layout)

        def enqueue_storage_to_compute(plan, slot, context, *, prepare):
            """Align swap prefetch/offload with one FlexShard redistribution bucket."""
            nonlocal deferred_d2h_names
            if deferred_d2h_names is not None:
                raise RuntimeError("Nested FlexShard storage-to-compute enqueue is unsupported")

            prefetch_plan(plan)
            deferred_d2h_names = []
            try:
                work = original_enqueue_storage_to_compute(
                    plan,
                    slot,
                    context,
                    prepare=prepare,
                )
            except Exception:
                deferred_d2h_names = None
                raise

            completed_d2h_names = deferred_d2h_names
            deferred_d2h_names = None

            with torch_npu.npu.stream(context.transfer_stream):
                for name in completed_d2h_names:
                    swap_api.execute(name, "D2H")
            return work

        def prepare_local(optimizer_self, compute_layout, out) -> None:
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            momentum = optimizer_self.state[compute_layout.param]["momentum_buffer"]
            local = momentum.to_local() if isinstance(momentum, DTensor) else momentum
            if local.numel() == 0:
                original_prepare_local(compute_layout, out)
                return
            submit_h2d(compute_layout)
            swap_api.execute(name, "WAIT_DEVICE")
            original_prepare_local(compute_layout, out)

            if deferred_d2h_names is None:
                swap_api.execute(name, "D2H")
            else:
                deferred_d2h_names.append(name)

        optimizer._momentum = MethodType(momentum, optimizer)
        optimizer._prepare_local = MethodType(prepare_local, optimizer)
        runtime._enqueue_storage_to_compute = enqueue_storage_to_compute

    @staticmethod
    def _swap_adamw(optimizer: torch.optim.AdamW) -> None:
        swap = _NovaSwapAdamW(optimizer)

        @Optimizer.profile_hook_step
        @_use_grad_for_differentiable
        def step(_optimizer, closure=None):
            return swap.step(closure)

        # Install the intentional instance-level AdamW bucket-swap wrapper.
        optimizer.step = MethodType(step, optimizer)  # pyrefly: ignore [bad-override]

    def state_dict(self) -> dict[str, Any]:
        raise RuntimeError("Optimizer state swap v1 does not support optimizer checkpoint save")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise RuntimeError("Optimizer state swap v1 does not support optimizer checkpoint load")


@override(
    target=OptimizersContainer.Config,
    description="Offload DistMuon and AdamW state by globally unique names",
)
def swap_optimizer(
    cfg: OptimizersContainer.Config,
) -> OptimizerStateSwapContainer.Config:
    return derive(cfg, OptimizerStateSwapContainer.Config)
