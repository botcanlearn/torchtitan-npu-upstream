# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""CPU contracts for the explicit optimizer-state swap override policy."""

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, OverrideConfig, apply_overrides

from torchtitan_npu.override.common import optimizer as product_swap
from torchtitan_npu.override.common.optimizer import OptimizerStateSwapContainer

pytestmark = pytest.mark.cpu


def _patch_container_global(monkeypatch, name: str, value) -> None:
    """Patch the module globals used by the imported container class.

    Other CPU tests can temporarily replace ``torchtitan_npu`` modules. Patch
    the class under test directly so fakes remain effective even when a package
    attribute points at a different module instance.
    """
    monkeypatch.setitem(
        OptimizerStateSwapContainer._swap_adamw.__globals__,
        name,
        value,
    )


def _patch_container_swap_api(monkeypatch, swap_api) -> None:
    _patch_container_global(monkeypatch, "swap_api", swap_api)


class _Root(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )


def test_optimizer_state_swap_override_replaces_optimizer_config() -> None:
    config = _Root.Config()

    replacements = apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.optimizer.swap_optimizer",
            ]
        ),
        config,
    )

    assert len(replacements) == 1
    assert isinstance(config.optimizer, OptimizerStateSwapContainer.Config)


def test_virtual_optimizer_override_replaces_optimizer_config() -> None:
    config = _Root.Config()

    replacements = apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.optimizer.virtual",
            ]
        ),
        config,
    )

    assert len(replacements) == 1
    assert type(config.optimizer).__module__ == "torchtitan_npu.override.common.optimizer"
    assert type(config.optimizer).__qualname__ == "VirtualOptimizersContainer.Config"


def test_virtual_optimizers_container_registers_state_init_hook(monkeypatch) -> None:
    hooks = []
    optimizer = SimpleNamespace(register_step_pre_hook=hooks.append)

    def initialize_container(self, *, config, model_parts):
        self.optimizers = [optimizer]

    monkeypatch.setattr(
        product_swap.OptimizersContainer,
        "__init__",
        initialize_container,
    )

    product_swap.VirtualOptimizersContainer(config=object(), model_parts=[])

    assert hooks == [product_swap._swap_state_init_hook]


def test_optimizer_state_swap_conflicts_with_virtual_optimizer_override() -> None:
    config = _Root.Config()

    with pytest.raises(ValueError, match="both claim node 'optimizer'"):
        apply_overrides(
            OverrideConfig(
                imports=[
                    "torchtitan_npu.override.common.optimizer.virtual",
                    "torchtitan_npu.override.common.optimizer.swap_optimizer",
                ]
            ),
            config,
        )


def test_optimizer_state_swap_rejects_optimizer_state_dict_access() -> None:
    optimizers = object.__new__(OptimizerStateSwapContainer)
    with pytest.raises(RuntimeError, match="does not support optimizer checkpoint save"):
        optimizers.state_dict()
    with pytest.raises(RuntimeError, match="does not support optimizer checkpoint load"):
        optimizers.load_state_dict({})


def test_optimizer_state_swap_has_no_unowned_cleanup_facade() -> None:
    source = product_swap.__file__
    assert source is not None
    contents = Path(source).read_text()

    assert "close_swap_state" not in contents
    assert "_torchtitan_npu_close_swap_state" not in contents
    assert "_torchtitan_npu_swap_error" not in contents


def test_optimizer_state_swap_adamw_pipelines_novaswap_buckets(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(4))
    parameter_b = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter_a, parameter_b], lr=1e-3, foreach=False)
    events = []
    phases = {}

    def register_tensor(tensor, name) -> None:
        events.append(("register", name, tensor))

    def execute(name, action) -> None:
        events.append(("execute", name, action))
        if action == "D2H":
            phases[name] = "D2H"

    fake_swap_api = SimpleNamespace(
        register_tensor=register_tensor,
        execute=execute,
        get_handle_phase=lambda name: phases.get(name),
        remove_tensor=lambda name: events.append(("remove", name)),
    )
    _patch_container_swap_api(monkeypatch, fake_swap_api)
    OptimizerStateSwapContainer._swap_adamw(optimizer)

    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    optimizer.step()

    names = [
        f"adamw.{id(optimizer)}.bucket.0",
        f"adamw.{id(optimizer)}.bucket.1",
    ]
    assert [event[1] for event in events if event[0] == "register"] == names
    assert [event[2] for event in events if event[0] == "execute"] == [
        "D2H",
        "D2H",
    ]
    for parameter in (parameter_a, parameter_b):
        state = optimizer.state[parameter]
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
        assert state["exp_avg"].untyped_storage().data_ptr() == state["exp_avg_sq"].untyped_storage().data_ptr()
    assert "step" not in optimizer.param_groups[0]

    events.clear()
    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    optimizer.step()

    assert [event[2] for event in events if event[0] == "execute"] == [
        "H2D",
        "WAIT_DEVICE",
        "H2D",
        "D2H",
        "WAIT_DEVICE",
        "D2H",
    ]


def test_optimizer_state_swap_adamw_prefetches_before_current_bucket_compute(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(4))
    parameter_b = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter_a, parameter_b], lr=1e-3, foreach=False)
    events = []
    monkeypatch.setattr(
        product_swap,
        "swap_api",
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: events.append((action, name)),
        ),
    )
    swap = product_swap._NovaSwapAdamW(optimizer)

    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    swap.step()

    events.clear()
    swap._original_step = lambda closure=None: events.append(("AdamW", None))
    swap.step()

    assert [action for action, _ in events] == [
        "H2D",
        "WAIT_DEVICE",
        "H2D",
        "AdamW",
        "D2H",
        "WAIT_DEVICE",
        "AdamW",
        "D2H",
    ]


def test_optimizer_state_swap_adamw_keeps_stock_state_and_updates(monkeypatch) -> None:
    swapped_parameters = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0])),
        torch.nn.Parameter(torch.tensor([3.0, 4.0])),
    ]
    reference_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in swapped_parameters
    ]
    swapped = torch.optim.AdamW(swapped_parameters, lr=1e-3, foreach=False)
    reference = torch.optim.AdamW(reference_parameters, lr=1e-3, foreach=False)
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: None,
        ),
    )
    OptimizerStateSwapContainer._swap_adamw(swapped)

    for _ in range(2):
        original_param_groups = swapped.param_groups
        for swapped_parameter, reference_parameter in zip(
            swapped_parameters, reference_parameters, strict=True
        ):
            grad = torch.full_like(swapped_parameter, 0.25)
            swapped_parameter.grad = grad
            reference_parameter.grad = grad.clone()
        swapped.step()
        reference.step()
        assert swapped.param_groups is original_param_groups

    for swapped_parameter, reference_parameter in zip(
        swapped_parameters, reference_parameters, strict=True
    ):
        assert torch.equal(swapped_parameter, reference_parameter)
        swapped_state = swapped.state[swapped_parameter]
        reference_state = reference.state[reference_parameter]
        assert swapped_state.keys() == reference_state.keys()
        for key in swapped_state:
            assert torch.equal(swapped_state[key], reference_state[key])


def test_optimizer_state_swap_keeps_tensor_wait_and_local_offload_at_prepare(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(2))
    parameter_b = torch.nn.Parameter(torch.ones(3))
    layout_a = SimpleNamespace(param=parameter_a, fqn="a")
    layout_b = SimpleNamespace(param=parameter_b, fqn="b")
    events = []

    class FakeMuon:
        def __init__(self) -> None:
            self.state = {
                parameter_a: {"momentum_buffer": parameter_a.detach().clone()},
                parameter_b: {"momentum_buffer": parameter_b.detach().clone()},
            }

            self._redistribution_runtime = SimpleNamespace(
                _enqueue_storage_to_compute=lambda *args, **kwargs: None,
            )

        def _momentum(self, compute_layout, grad):
            return self.state[compute_layout.param]["momentum_buffer"]

        def _prepare_local(self, compute_layout, out):
            events.append(("prepare", compute_layout))

    def execute(name, action) -> None:
        events.append(("execute", name, action))

    phases = {}

    def get_handle_phase(name):
        return phases.get(name, "D2H")

    def execute_with_phase(name, action) -> None:
        execute(name, action)
        if action == "H2D":
            phases[name] = "H2D"
        elif action == "D2H":
            phases[name] = "D2H"

    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=execute_with_phase,
            get_handle_phase=get_handle_phase,
        ),
    )

    optimizer = FakeMuon()
    OptimizerStateSwapContainer._swap_muon(optimizer, model_part=0)

    optimizer._prepare_local(layout_a, None)
    optimizer._prepare_local(layout_b, None)

    assert [event[2] for event in events if event[0] == "execute"] == [
        "H2D",
        "WAIT_DEVICE",
        "D2H",
        "H2D",
        "WAIT_DEVICE",
        "D2H",
    ]
    assert [event[0] for event in events] == [
        "execute",
        "execute",
        "prepare",
        "execute",
        "execute",
        "execute",
        "prepare",
        "execute",
    ]


def test_optimizer_state_swap_prefetches_plan_before_a2a_and_defers_offload(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(2))
    parameter_b = torch.nn.Parameter(torch.ones(3))
    parameter_c = torch.nn.Parameter(torch.ones(4))
    layout_a = SimpleNamespace(param=parameter_a, fqn="a")
    layout_b = SimpleNamespace(param=parameter_b, fqn="b")
    layout_c = SimpleNamespace(param=parameter_c, fqn="c")
    events = []
    phases = {}
    streams = []

    class FakeRuntime:
        def _enqueue_storage_to_compute(self, plan, slot, context, *, prepare):
            for layout in plan.redistributed_items:
                prepare(layout, None)
            events.append(("a2a", plan))
            return "work"

    class FakeMuon:
        def __init__(self) -> None:
            self.state = {
                parameter_a: {"momentum_buffer": parameter_a.detach().clone()},
                parameter_b: {"momentum_buffer": parameter_b.detach().clone()},
                parameter_c: {"momentum_buffer": parameter_c.detach().clone()},
            }
            self._redistribution_runtime = FakeRuntime()

        def _momentum(self, compute_layout, grad):
            return self.state[compute_layout.param]["momentum_buffer"]

        def _prepare_local(self, compute_layout, out):
            events.append(("prepare", compute_layout.fqn))

    def execute(name, action) -> None:
        events.append((action, name))
        phases[name] = action

    def stream_context(stream):
        streams.append(stream)
        return nullcontext()

    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=execute,
            get_handle_phase=lambda name: phases.get(name, "D2H"),
        ),
    )
    _patch_container_global(
        monkeypatch,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(stream=stream_context)),
    )

    optimizer = FakeMuon()
    OptimizerStateSwapContainer._swap_muon(optimizer, model_part=0)
    plan = SimpleNamespace(
        redistributed_items=(layout_a, layout_b),
        unredistributed_items=(layout_c,),
    )

    transfer_stream = object()
    assert optimizer._redistribution_runtime._enqueue_storage_to_compute(
        plan,
        slot=None,
        context=SimpleNamespace(transfer_stream=transfer_stream),
        prepare=optimizer._prepare_local,
    ) == "work"
    assert streams == [transfer_stream]
    assert [event[0] for event in events] == [
        "H2D",
        "H2D",
        "H2D",
        "WAIT_DEVICE",
        "prepare",
        "WAIT_DEVICE",
        "prepare",
        "a2a",
        "D2H",
        "D2H",
    ]
    assert [event[1] for event in events[:3]] == [
        product_swap.make_swap_state_name("optimizer_state", 0, "a", "momentum_buffer"),
        product_swap.make_swap_state_name("optimizer_state", 0, "b", "momentum_buffer"),
        product_swap.make_swap_state_name("optimizer_state", 0, "c", "momentum_buffer"),
    ]
