# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for DeepSeek-V4 Context Parallel implementation.

Covers ``_allgather_seq``, ``_WindowExchange`` autograd function, and
``_detect_dsv4`` / ``_apply_dsv4`` detection logic.

Run with::

    pytest tests/unit_tests/distributed/ -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed._functional_collectives as ft_c
from torch.distributed.device_mesh import init_device_mesh

from torchtitan_npu.distributed.context_parallel.compressor_attention_cp import (
    _allgather_seq,
    _detect_dsv4,
    _WindowExchange,
)


def _make_cpu_mesh():
    """Create a single-rank CPU device mesh using the current process group."""
    return init_device_mesh("cpu", (1,))


# ===========================================================================
# _allgather_seq — single-rank
# ===========================================================================


@pytest.mark.usefixtures("single_rank_process_group")
class TestAllgatherSeq:
    """Single-rank tests for ``_allgather_seq`` using gloo identity semantics."""

    @staticmethod
    def test_preserves_requires_grad():
        mesh = _make_cpu_mesh()
        x = torch.randn(2, 128, 64, device="cpu", requires_grad=True)

        gathered = _allgather_seq(x, mesh)

        assert gathered.requires_grad, (
            "_allgather_seq must preserve requires_grad; calling .wait() on AsyncCollectiveTensor would detach it"
        )

    @staticmethod
    def test_gradient_flows():
        mesh = _make_cpu_mesh()
        x = torch.randn(2, 128, 64, device="cpu", requires_grad=True)

        gathered = _allgather_seq(x, mesh)
        gathered.sum().backward()

        assert x.grad is not None, "Gradient should flow back through _allgather_seq"
        assert torch.allclose(x.grad, torch.ones_like(x)), "grad of allgather+sum should be all-ones"

    @staticmethod
    def test_returns_plain_tensor():
        mesh = _make_cpu_mesh()
        x = torch.randn(2, 128, 64, device="cpu")

        gathered = _allgather_seq(x, mesh)

        assert type(gathered) is torch.Tensor, (
            "_allgather_seq must return a plain torch.Tensor, not DTensor or AsyncCollectiveTensor"
        )
        assert not isinstance(gathered, ft_c.AsyncCollectiveTensor)


# ===========================================================================
# _WindowExchange — single-rank (identity path)
# ===========================================================================


@pytest.mark.usefixtures("single_rank_process_group")
class TestWindowExchange:
    """Single-rank tests for ``_WindowExchange`` (world_size=1 identity path)."""

    @staticmethod
    def test_forward_identity():
        mesh = _make_cpu_mesh()
        group = mesh.get_group()
        tensor = torch.randn(2, 32, 4, device="cpu", requires_grad=True)

        result = _WindowExchange.apply(tensor, 8, group)

        assert result.shape == tensor.shape
        assert torch.allclose(result, tensor)

    @staticmethod
    def test_backward_identity():
        mesh = _make_cpu_mesh()
        group = mesh.get_group()
        tensor = torch.randn(2, 32, 4, device="cpu", requires_grad=True)

        result = _WindowExchange.apply(tensor, 8, group)
        loss = result.sum()
        loss.backward()

        assert tensor.grad is not None
        assert torch.allclose(tensor.grad, torch.ones_like(tensor))

    @staticmethod
    def test_gradcheck():
        mesh = _make_cpu_mesh()
        group = mesh.get_group()
        window = 4

        x = torch.randn(2, 16, 2, dtype=torch.float64, device="cpu", requires_grad=True)

        def forward_fn(t):
            return _WindowExchange.apply(t, window, group)

        torch.autograd.gradcheck(forward_fn, x)

    @staticmethod
    def test_backward_nondiff_args_none():
        mesh = _make_cpu_mesh()
        group = mesh.get_group()

        ctx = MagicMock()
        ctx.rank = 0
        ctx.world_size = 1
        ctx.group = group
        ctx.window = 4
        ctx.forward_sent = False
        ctx.forward_recvd = False

        grad_output = torch.randn(2, 32, 4)
        grad_tensor, grad_window, grad_group = _WindowExchange.backward(ctx, grad_output)

        assert torch.allclose(grad_tensor, grad_output)
        assert grad_window is None
        assert grad_group is None


# ===========================================================================
# _detect_dsv4 — real model class detection
# ===========================================================================


class TestDetectDSv4:
    """Tests for ``_detect_dsv4`` against real model classes."""

    @staticmethod
    def test_matches_v4_attention():
        try:
            from torchtitan_npu.models.deepseek_v4.model import (
                Attention,
                DeepSeekV4Model,
            )
        except ImportError:
            pytest.skip("DeepSeek V4 model not available")

        with torch.device("meta"):
            config = DeepSeekV4Model.Config()
            attn = Attention.Config(layer_id=0, args=config).build()

        assert _detect_dsv4(attn) is True, "DS V4 Attention must be detected: has compress_ratio + pre_attention"
        assert hasattr(attn, "compress_ratio")
        assert hasattr(attn, "pre_attention")

    @staticmethod
    def test_rejects_v32_attention():
        """V32 Attention has ``pre_attention`` but NOT ``compress_ratio``.

        V3 Config requires explicit Linear/RNNorm Config fields, so meta-device
        instantiation is complex.  We test the detector logic directly: the
        detector checks ``hasattr(module, "compress_ratio")``, and V32's
        ``Attention.__init__`` never sets that attribute.
        """
        mock = SimpleNamespace(pre_attention=MagicMock())

        assert _detect_dsv4(mock) is False, (
            "Module with pre_attention but without compress_ratio must not be detected (this is the V32 case)"
        )
        assert hasattr(mock, "pre_attention")
        assert not hasattr(mock, "compress_ratio")

    @staticmethod
    def test_rejects_v3_attention():
        """V3 Attention has neither ``pre_attention`` nor ``compress_ratio``."""
        try:
            from torchtitan.models.common.linear import Linear
            from torchtitan.models.common.rmsnorm import RMSNorm
            from torchtitan.models.deepseek_v3.model import (
                Attention as DeepSeekV3Attention,
            )
        except ImportError:
            pytest.skip("DeepSeek V3 model (torchtitan) not available")

        with torch.device("meta"):
            cfg = DeepSeekV3Attention.Config(
                n_heads=64,
                dim=4096,
                wq=Linear.Config(in_features=4096, out_features=12288),
                wkv_a=Linear.Config(in_features=4096, out_features=576),
                wkv_b=Linear.Config(in_features=512, out_features=16384),
                wo=Linear.Config(in_features=8192, out_features=4096),
                q_norm=RMSNorm.Config(normalized_shape=0),
                kv_norm=RMSNorm.Config(normalized_shape=512),
                q_lora_rank=0,
            )
            attn = cfg.build()

        assert _detect_dsv4(attn) is False, (
            "DS V3 Attention must not be detected: lacks both compress_ratio and pre_attention"
        )
        assert not hasattr(attn, "compress_ratio")
        assert not hasattr(attn, "pre_attention")
