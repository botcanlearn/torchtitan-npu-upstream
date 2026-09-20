# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overrides for NPU swap-backed optimizer states and their checkpoints."""

from collections import deque
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch_npu
from torch.distributed._tensor import DTensor
from torch.optim.optimizer import Optimizer, _get_scalar_dtype, _use_grad_for_differentiable
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import derive, override
from torchtitan.distributed.flex_shard.dist_muon import DistMuon

from torchtitan_npu.extensions.components.optimizer import HostSparseOptimizersContainer
from torchtitan_npu.extensions.novaswap import swap_api

_ADAMW_SWAP_BUCKET_TIMES = 16
_ADAMW_SWAP_RESIDENT_TARGET_BUCKETS = 3


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


@dataclass
class _AdamWSwapBucket:
    group: dict[str, Any]
    parameters: tuple[torch.Tensor, ...]
    state_name: str
    state_numel: int
    state_nbytes: int
    flat: torch.Tensor | None = None


class _NovaSwapAdamW:
    """Run stock AdamW state through NovaSwap with a bucket-level pipeline."""

    def __init__(self, optimizer: torch.optim.AdamW) -> None:
        self.optimizer = optimizer
        self._original_step = optimizer.step
        self._buckets: list[_AdamWSwapBucket] = []
        self._target_bucket_bytes = 1
        self._largest_bucket_bytes = 1

    @staticmethod
    def _submit(bucket: _AdamWSwapBucket, action: str) -> None:
        swap_api.execute(bucket.state_name, action)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        new_buckets = self._plan_uninitialized_buckets()
        device_budget = _ADAMW_SWAP_RESIDENT_TARGET_BUCKETS * max(
            self._target_bucket_bytes,
            self._largest_bucket_bytes,
        )
        pending_releases: deque[_AdamWSwapBucket] = deque()
        resident_bytes = 0

        def reserve(bucket: _AdamWSwapBucket) -> None:
            nonlocal resident_bytes
            while pending_releases and resident_bytes + bucket.state_nbytes > device_budget:
                oldest = pending_releases.popleft()
                swap_api.wait_for_device_release(oldest.state_name)
                resident_bytes -= oldest.state_nbytes
            if resident_bytes + bucket.state_nbytes > device_budget:
                raise RuntimeError("AdamW swap resident-state accounting exceeded the device budget")
            resident_bytes += bucket.state_nbytes

        def submit_h2d(bucket: _AdamWSwapBucket) -> None:
            reserve(bucket)
            swap_api.wait_for_device_release(bucket.state_name)
            self._submit(bucket, "H2D")

        for bucket in new_buckets:
            reserve(bucket)
            self._initialize_bucket_state(bucket)
            self._update_bucket(bucket)
            self._submit(bucket, "D2H")
            pending_releases.append(bucket)
        self._buckets.extend(new_buckets)

        new_bucket_ids = {id(bucket) for bucket in new_buckets}
        active_buckets = [
            bucket
            for bucket in self._buckets
            if id(bucket) not in new_bucket_ids and any(parameter.grad is not None for parameter in bucket.parameters)
        ]
        if not active_buckets:
            return loss

        submit_h2d(active_buckets[0])
        for bucket_index, bucket in enumerate(active_buckets):
            self._submit(bucket, "WAIT_DEVICE")
            if bucket_index + 1 < len(active_buckets):
                submit_h2d(active_buckets[bucket_index + 1])
            self._update_bucket(bucket)
            self._submit(bucket, "D2H")
            pending_releases.append(bucket)
        return loss

    def _plan_uninitialized_buckets(self) -> list[_AdamWSwapBucket]:
        """Plan only states that PyTorch AdamW would lazily create this step."""
        uninitialized = [
            (group, parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None and not self.optimizer.state[parameter]
        ]
        if not uninitialized:
            return []

        total_state_bytes = sum(
            self._state_numel(group, parameter) * _local_tensor(parameter).element_size()
            for group, parameter in uninitialized
        )
        bucket_byte_limit = max(total_state_bytes // _ADAMW_SWAP_BUCKET_TIMES, 1)
        self._target_bucket_bytes = max(self._target_bucket_bytes, bucket_byte_limit)
        buckets: list[_AdamWSwapBucket] = []
        current_group: dict[str, Any] | None = None
        current_parameters: list[torch.Tensor] = []
        current_state_bytes = 0
        current_signature: tuple[torch.device, torch.dtype] | None = None

        def append_bucket() -> None:
            if not current_parameters:
                return
            if current_group is None:
                raise RuntimeError("AdamW swap bucket has no parameter group")
            state_numel = sum(self._state_numel(current_group, parameter) for parameter in current_parameters)
            state_nbytes = state_numel * _local_tensor(current_parameters[0]).element_size()
            self._largest_bucket_bytes = max(self._largest_bucket_bytes, state_nbytes)
            buckets.append(
                _AdamWSwapBucket(
                    group=current_group,
                    parameters=tuple(current_parameters),
                    state_name=make_swap_state_name(
                        "adamw", id(self.optimizer), "bucket", len(self._buckets) + len(buckets)
                    ),
                    state_numel=state_numel,
                    state_nbytes=state_nbytes,
                )
            )

        for group, parameter in uninitialized:
            local_parameter = _local_tensor(parameter)
            parameter_signature = (local_parameter.device, local_parameter.dtype)
            parameter_state_bytes = self._state_numel(group, parameter) * local_parameter.element_size()
            if current_parameters:
                same_group = group is current_group
                fits_bucket = current_state_bytes + parameter_state_bytes <= bucket_byte_limit
                same_signature = parameter_signature == current_signature
                if not (same_group and fits_bucket and same_signature):
                    append_bucket()
                    current_parameters = []
                    current_state_bytes = 0
            current_group = group
            current_signature = parameter_signature
            current_parameters.append(parameter)
            current_state_bytes += parameter_state_bytes
        append_bucket()
        return buckets

    @staticmethod
    def _state_keys(group: dict[str, Any]) -> tuple[str, ...]:
        if group["amsgrad"]:
            return "exp_avg", "exp_avg_sq", "max_exp_avg_sq"
        return "exp_avg", "exp_avg_sq"

    def _state_numel(self, group: dict[str, Any], parameter: torch.Tensor) -> int:
        return len(self._state_keys(group)) * _local_tensor(parameter).numel()

    @staticmethod
    def _initialize_step(group: dict[str, Any], device: torch.device) -> torch.Tensor:
        if group["differentiable"]:
            raise ValueError("AdamW NovaSwap only supports differentiable=False")
        if group["foreach"]:
            raise ValueError("AdamW NovaSwap only supports foreach=False")
        # Match AdamW._init_group: fused/capturable paths keep step on-device;
        # otherwise the scalar remains on CPU. Only the moment allocation is
        # replaced by the bucket flat storage above.
        if group["capturable"] or group["fused"]:
            return torch.zeros(
                (),
                dtype=_get_scalar_dtype(is_fused=group["fused"]),
                device=device,
            )
        return torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")

    def _initialize_bucket_state(self, bucket: _AdamWSwapBucket) -> None:
        """Install flat NPU state views before the first stock AdamW update."""
        if bucket.flat is not None:
            raise RuntimeError(f"AdamW swap bucket {bucket.state_name!r} is already initialized")
        first_local = _local_tensor(bucket.parameters[0])
        flat = torch.zeros(bucket.state_numel, dtype=first_local.dtype, device=first_local.device)
        offset = 0
        state_keys = self._state_keys(bucket.group)
        for parameter in bucket.parameters:
            state = self.optimizer.state[parameter]
            if state:
                raise RuntimeError("AdamW swap attempted to initialize an existing optimizer state")
            local_parameter = _local_tensor(parameter)
            if (local_parameter.device, local_parameter.dtype) != (first_local.device, first_local.dtype):
                raise RuntimeError("AdamW swap bucket parameters must share a device and dtype")
            state["step"] = self._initialize_step(bucket.group, first_local.device)
            for state_key in state_keys:
                view = flat.narrow(0, offset, local_parameter.numel()).view_as(local_parameter)
                state[state_key] = _replace_local_tensor(parameter, view)
                offset += local_parameter.numel()
        if offset != bucket.state_numel:
            raise RuntimeError("AdamW swap bucket state size does not match its planned layout")
        bucket.flat = flat
        swap_api.register_tensor(flat, bucket.state_name)

    def _update_bucket(self, bucket: _AdamWSwapBucket) -> None:
        original_param_groups = self.optimizer.param_groups
        pre_hooks = self.optimizer._optimizer_step_pre_hooks
        post_hooks = self.optimizer._optimizer_step_post_hooks
        saved_pre_hooks = pre_hooks.copy()
        saved_post_hooks = post_hooks.copy()
        self.optimizer.param_groups = [{**bucket.group, "params": bucket.parameters}]
        # The public, instance-level wrapper retains PyTorch's step hooks once
        # per logical optimizer step. Calling the saved stock step per bucket
        # must not replay those external hooks for every bucket.
        pre_hooks.clear()
        post_hooks.clear()
        try:
            self._original_step()
        finally:
            self.optimizer.param_groups = original_param_groups
            pre_hooks.update(saved_pre_hooks)
            post_hooks.update(saved_post_hooks)


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


def _initialize_swap_state(optimizer, *, only_with_grad=True):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if only_with_grad and p.grad is None:
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
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = _make_swap(p).zero_()


def _swap_state_init_hook(optimizer, args, kwargs):
    _initialize_swap_state(optimizer)


class VirtualOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts):
        super().__init__(config=config, model_parts=model_parts)
        for opt in self._swap_optimizers():
            opt.register_step_pre_hook(_swap_state_init_hook)

    def _swap_optimizers(self):
        return (opt for opt in self.optimizers if isinstance(opt, (torch.optim.Adam, torch.optim.AdamW)))

    def state_dict(self):
        # DCP also needs state for parameters that were unused in training.
        for opt in self._swap_optimizers():
            _initialize_swap_state(opt, only_with_grad=False)
        return super().state_dict()

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict may replace the moment tensors. Preserve
        # swap-backed destinations even when loading ordinary checkpoint tensors.
        destinations = []
        for opt in self._swap_optimizers():
            _initialize_swap_state(opt, only_with_grad=False)
            for group in opt.param_groups:
                for param in group["params"]:
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in opt.state[param]:
                            destinations.append((opt, param, key, opt.state[param][key]))
        super().load_state_dict(state_dict)
        with torch.no_grad():
            for opt, param, key, destination in destinations:
                destination.copy_(opt.state[param][key])
                opt.state[param][key] = destination


class VirtualHostSparseOptimizersContainer(VirtualOptimizersContainer, HostSparseOptimizersContainer):
    """Swap dense Adam states while retaining Host sparse-table lifecycle hooks."""

    @dataclass(kw_only=True, slots=True)
    class Config(HostSparseOptimizersContainer.Config, VirtualOptimizersContainer.Config):
        pass


@override(
    target=OptimizersContainer.Config,
    description="Allocate Adam/AdamW states in swap memory (host-offload)",
)
def virtual(
    cfg: OptimizersContainer.Config,
) -> VirtualOptimizersContainer.Config | VirtualHostSparseOptimizersContainer.Config:
    if isinstance(cfg, HostSparseOptimizersContainer.Config):
        return derive(cfg, VirtualHostSparseOptimizersContainer.Config)
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
        if any(state for state in optimizer.state.values()):
            raise ValueError("AdamW NovaSwap must be installed before AdamW initializes optimizer state")
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


# Both branches inherit the same OptimizersContainer.step overloads; Python's
# MRO selects HostSparse's implementation before their common base.
class HostSparseOptimizerStateSwapContainer(  # pyrefly: ignore [inconsistent-inheritance]
    OptimizerStateSwapContainer, HostSparseOptimizersContainer
):
    """Apply the upstream dense-state swap while retaining Host sparse updates."""

    @dataclass(kw_only=True, slots=True)
    class Config(HostSparseOptimizersContainer.Config, OptimizerStateSwapContainer.Config):
        pass


@override(
    target=OptimizersContainer.Config,
    description="Offload DistMuon and AdamW state by globally unique names",
)
def swap_optimizer(
    cfg: OptimizersContainer.Config,
) -> OptimizerStateSwapContainer.Config | HostSparseOptimizerStateSwapContainer.Config:
    if isinstance(cfg, HostSparseOptimizersContainer.Config):
        return derive(cfg, HostSparseOptimizerStateSwapContainer.Config)
    return derive(cfg, OptimizerStateSwapContainer.Config)
