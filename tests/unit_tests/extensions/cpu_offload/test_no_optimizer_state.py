# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU-resident optimizer state under ``--training.enable-cpu-offload``.

``swap_optimizer`` stays the single toggle: listed -> CPU-canonical optimizer
state (staged per step); absent -> the moments live and update on the compute
device with no per-step round trip.
"""

import pytest
import torch
import torch_npu

from torchtitan_npu.config.configs import OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.cpu_offload.cpu_offload_adamw import CpuOffloadAdamW
from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging
from torchtitan_npu.extensions.trainer import TrainerEx

requires_npu = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU not available"
)


def _step_sequence(
    offload_states: bool, grads: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, set[int], str]:
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(4, 6))
    device = torch.device("npu", torch_npu.npu.current_device())
    staging = CpuStaging(device, owner="test-no-opt-state")
    optimizer = CpuOffloadAdamW(
        [parameter],
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        staging=staging,
        offload_states=offload_states,
    )
    for gradient in grads:
        parameter.grad = gradient.clone()
        optimizer.step()
    optimizer._staging.wait()
    staged_kinds = {key[1] for key in optimizer._work_buffers}
    exp_avg = optimizer.state[parameter]["exp_avg"].detach()
    placement = exp_avg.device.type
    exp_avg_value = exp_avg.cpu().clone()
    staging.close()
    return parameter.detach().clone(), exp_avg_value, staged_kinds, placement


@requires_npu
def test_npu_resident_state_matches_cpu_canonical_updates() -> None:
    torch.manual_seed(1)
    grads = [torch.randn(4, 6) for _ in range(3)]
    cpu_parameter, cpu_exp_avg, _, _ = _step_sequence(True, grads)
    npu_parameter, npu_exp_avg, _, _ = _step_sequence(False, grads)
    torch.testing.assert_close(npu_parameter, cpu_parameter)
    torch.testing.assert_close(npu_exp_avg, cpu_exp_avg)


@requires_npu
def test_npu_resident_state_placement_and_staging_planes() -> None:
    torch.manual_seed(2)
    grads = [torch.randn(4, 6) for _ in range(2)]

    _, _, cpu_kinds, cpu_placement = _step_sequence(True, grads)
    assert cpu_placement == "cpu"
    assert cpu_kinds == {0, 1, 2, 3}

    _, _, npu_kinds, npu_placement = _step_sequence(False, grads)
    assert npu_placement == "npu"
    # Only the parameter (0) and gradient (1) planes are staged; the moments
    # update in place instead of round-tripping through work buffers.
    assert npu_kinds == {0, 1}


def test_enable_cpu_offload_defaults_to_npu_resident_state(monkeypatch) -> None:
    from dataclasses import replace

    from torchtitan_npu.extensions import trainer as trainer_module
    from torchtitan_npu.override.common.optimizer import (
        CpuOffloadNpuStateOptimizersContainer,
        CpuOffloadOptimizersContainer,
        swap_optimizer,
    )
    from torchtitan_npu.patches.torch_npu import cpu_dtensor_init

    class ReachedUpstream(Exception):
        pass

    captured: dict[str, object] = {}

    def capture_runtime(self, runtime) -> None:
        captured["optimizer"] = runtime.optimizer
        raise ReachedUpstream

    monkeypatch.setattr(trainer_module, "set_allow_hf32", lambda _: None)
    monkeypatch.setattr(cpu_dtensor_init, "install", lambda: None)
    monkeypatch.setattr(trainer_module.Trainer, "__init__", capture_runtime)

    config = TrainerEx.Config(training=TrainingConfig(enable_cpu_offload=True))
    # Config rebuilds keep the plain schema: the derive happens exactly once,
    # at trainer construction (regression: config.build() does replace(self),
    # which re-runs __post_init__ on the already-derived node).
    for cfg in (config, replace(config)):
        assert type(cfg.optimizer) is OptimizerConfig
        assert isinstance(cfg.optimizer.materialize(), type(None))

    with pytest.raises(ReachedUpstream):
        TrainerEx(config)
    runtime_optimizer = captured["optimizer"]
    assert isinstance(runtime_optimizer, CpuOffloadNpuStateOptimizersContainer.Config)
    assert runtime_optimizer._cpu_offload is True

    # Listing swap_optimizer re-derives to the CPU-canonical-state container.
    derived = swap_optimizer(runtime_optimizer)
    assert isinstance(derived, CpuOffloadOptimizersContainer.Config)
    assert not isinstance(derived, CpuOffloadNpuStateOptimizersContainer.Config)

    # Without the offload switch the optimizer schema stays untouched.
    plain = TrainerEx.Config()
    assert type(plain.optimizer) is OptimizerConfig
