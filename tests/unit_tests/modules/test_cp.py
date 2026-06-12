# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from torch.distributed._tensor import DTensor
from torchtitan.models.common.attention import ScaledDotProductAttention

from torchtitan_npu.distributed.context_parallel.ulysses_cp import (
    AllToAll,
    UlyssesCP,
    all_to_all,
)


def _make_cpu_mesh():
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,))


# ---------------------------------------------------------------------------
# Ulysses CP tests
# ---------------------------------------------------------------------------


def _make_cpu_mesh_ulysses():
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,))


@pytest.fixture
def mock_all_to_all_identity_for_gloo():
    def _fake_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        assert len(output_tensor_list) == len(input_tensor_list)
        for out_t, in_t in zip(output_tensor_list, input_tensor_list, strict=True):
            out_t.copy_(in_t)
        return None

    with patch.object(torch.distributed, "all_to_all", side_effect=_fake_all_to_all):
        yield


@pytest.mark.usefixtures("single_rank_process_group", "mock_all_to_all_identity_for_gloo")
class TestAllToAll:
    @staticmethod
    def test_single_rank_preserves_shape_and_values():
        mesh = _make_cpu_mesh_ulysses()
        t = torch.randn(2, 4, 8, 16)

        result = all_to_all(t, mesh, scatter_dim=1, gather_dim=2)

        assert result.shape == t.shape
        assert torch.allclose(result, t)

    @staticmethod
    def test_output_is_plain_tensor():
        mesh = _make_cpu_mesh_ulysses()
        t = torch.randn(1, 4, 4, 8)

        result = all_to_all(t, mesh, scatter_dim=1, gather_dim=2)

        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, DTensor)

    @staticmethod
    def test_backward_grad_shape_matches_input():
        mesh = _make_cpu_mesh_ulysses()
        t = torch.randn(2, 4, 8, 16, requires_grad=True)

        output = AllToAll.apply(t, mesh, 1, 2)
        output.sum().backward()

        assert t.grad is not None
        assert t.grad.shape == t.shape

    @staticmethod
    def test_single_rank_backward_is_identity():
        mesh = _make_cpu_mesh_ulysses()
        t = torch.randn(2, 4, 8, 16, requires_grad=True)
        grad_ref = torch.ones(2, 4, 8, 16)

        output = AllToAll.apply(t, mesh, 1, 2)
        output.backward(torch.ones_like(output))

        assert torch.allclose(t.grad, grad_ref)


@pytest.mark.usefixtures("single_rank_process_group")
class TestUlyssesCPApply:
    @staticmethod
    def test_registers_pre_hook():
        module = ScaledDotProductAttention(
            ScaledDotProductAttention.Config()  # pyrefly: ignore [too-many-args]
        )
        mesh = _make_cpu_mesh()

        cp = UlyssesCP()
        cp._apply(module, mesh)

        assert len(module._forward_pre_hooks) > 0

    @staticmethod
    def test_registers_forward_hook():
        module = ScaledDotProductAttention(
            ScaledDotProductAttention.Config()  # pyrefly: ignore [too-many-args]
        )
        mesh = _make_cpu_mesh()

        cp = UlyssesCP()
        cp._apply(module, mesh)

        assert len(module._forward_hooks) > 0

    @staticmethod
    def test_rejects_wrong_module_type():
        import torch.nn as nn

        module = nn.Linear(4, 4)
        mesh = _make_cpu_mesh()

        cp = UlyssesCP()
        with pytest.raises(TypeError):
            cp._apply(module, mesh)
