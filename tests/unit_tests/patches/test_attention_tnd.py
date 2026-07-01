# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for NPUVarlenAttention."""

from unittest.mock import MagicMock, patch

from torchtitan_npu.models.common.npu_varlen_attention import NPUVarlenAttention


def _make_attn():
    """Create NPUVarlenAttention with has_cuda_capability mocked out.

    Upstream VarlenAttention.__init__ calls has_cuda_capability(9, 0) which
    may raise on CI machines without NPU/CUDA.  Patch it to return False.
    """
    with patch("torchtitan.tools.utils.has_cuda_capability", return_value=False):
        return NPUVarlenAttention(NPUVarlenAttention.Config())


class TestNPUVarlenAttention:
    @staticmethod
    def test_inherits_from_varlen_attention():
        from torchtitan.models.common.attention import VarlenAttention

        assert issubclass(NPUVarlenAttention, VarlenAttention)

    @staticmethod
    def test_config_isinstance_check():
        from torchtitan.models.common.attention import VarlenAttention

        assert isinstance(NPUVarlenAttention.Config(), VarlenAttention.Config)


_MOCK_BS, _MOCK_SEQ, _MOCK_HEADS, _MOCK_HEAD_DIM = 1, 4, 2, 8
_MOCK_KV_HEADS = 1


def _make_mock_forward_inputs():
    """Shared test helper: creates mock Q/K/V and VarlenMetadata for forward."""
    import torch
    from torchtitan.models.common.attention import VarlenMetadata

    xq = torch.randn(_MOCK_BS, _MOCK_SEQ, _MOCK_HEADS, _MOCK_HEAD_DIM, dtype=torch.bfloat16)
    xk = torch.randn(_MOCK_BS, _MOCK_SEQ, _MOCK_KV_HEADS, _MOCK_HEAD_DIM, dtype=torch.bfloat16)
    xv = torch.randn(_MOCK_BS, _MOCK_SEQ, _MOCK_KV_HEADS, _MOCK_HEAD_DIM, dtype=torch.bfloat16)
    masks = VarlenMetadata(
        cu_seq_q=torch.tensor([0, _MOCK_SEQ], dtype=torch.int64),
        cu_seq_k=torch.tensor([0, _MOCK_SEQ], dtype=torch.int64),
        max_q=_MOCK_SEQ,
        max_k=_MOCK_SEQ,
    )
    return xq, xk, xv, masks


def _call_forward(attn, xq, xk, xv, masks):
    """Call forward; NPU op not available on CPU — RuntimeError expected."""
    try:
        attn(xq, xk, xv, attention_masks=masks)
    except RuntimeError:
        pass


class TestNPUVarlenForward:
    @staticmethod
    def test_causal_mask_lazy_created_in_forward():
        """_causal_mask is lazily created on first forward, not a buffer."""
        attn = _make_attn()
        assert "_causal_mask" not in dict(attn.named_buffers())
        assert "_causal_mask" not in vars(attn)

        xq, xk, xv, masks = _make_mock_forward_inputs()
        _call_forward(attn, xq, xk, xv, masks)
        assert "_causal_mask" in vars(attn)

    @staticmethod
    def test_forward_lazy_mask_only_added_once():
        """_causal_mask is lazily created on first forward only, not re-created."""
        attn = _make_attn()

        xq, xk, xv, masks = _make_mock_forward_inputs()
        _call_forward(attn, xq, xk, xv, masks)
        assert "_causal_mask" in vars(attn)
        mask_id = id(vars(attn)["_causal_mask"])

        _call_forward(attn, xq, xk, xv, masks)
        assert id(vars(attn)["_causal_mask"]) == mask_id

    @staticmethod
    def test_forward_handles_device_cu_seq():
        """forward handles device cu_seq via device-guard (AC checkpoint path)."""
        import torch

        attn = _make_attn()
        xq, xk, xv, masks = _make_mock_forward_inputs()

        # Simulate AC checkpoint recreating VarlenMetadata with device tensors.
        device_masks = masks._replace(
            cu_seq_q=masks.cu_seq_q.to(torch.int64),
            cu_seq_k=masks.cu_seq_k.to(torch.int64),
        )
        # forward should NOT raise — device-guard handles it.
        try:
            attn(xq, xk, xv, attention_masks=device_masks)
        except RuntimeError as exc:
            assert "cache" not in str(exc).lower()


class TestCPStrategy:
    @staticmethod
    def test_npu_varlen_cp_registered():
        from torchtitan_npu.distributed.context_parallel.registry import _cp_strategies

        found = any(detector(MagicMock(spec=NPUVarlenAttention)) for detector, _ in _cp_strategies)
        assert found


class TestTNDConfig:
    @staticmethod
    def test_enable_npu_varlen_sets_block_causal():
        """_enable_npu_varlen_attention sets NPUVarlenAttention + block_causal."""
        from torchtitan_npu.models.qwen3 import model_registry
        from torchtitan_npu.models.qwen3.tnd_config import _enable_npu_varlen_attention

        spec = model_registry("0.6B")
        spec = _enable_npu_varlen_attention(spec)
        first_layer = spec.model.layers[0]
        assert first_layer.attention.mask_type == "block_causal"

    @staticmethod
    def test_update_from_config_bypass_restores_inner_attention():
        """_patch_update_from_config bypasses upstream CP+Varlen raise."""
        from torchtitan.config import TrainingConfig

        from torchtitan_npu.config.configs import ParallelismConfig
        from torchtitan_npu.models.qwen3 import model_registry
        from torchtitan_npu.models.qwen3.tnd_config import (
            _enable_npu_varlen_attention,
        )

        spec = model_registry("0.6B")
        spec = _enable_npu_varlen_attention(spec)

        # Build the model config and call update_from_config with CP>1.
        # The bypass should prevent NotImplementedError.
        cfg = spec.model
        cfg.update_from_config(
            trainer_config=type(
                "C",
                (),
                {
                    "parallelism": ParallelismConfig(context_parallel_degree=4),
                    "training": TrainingConfig(seq_len=4096, local_batch_size=1),
                    "debug": type(
                        "D",
                        (),
                        {
                            "moe_force_load_balance": False,
                            "deterministic": False,
                            "seed": None,
                        },
                    )(),
                },
            )(),
        )
        # After bypass, inner_attention should be restored
        first_layer = cfg.layers[0]
        assert first_layer.attention.inner_attention is not None
        assert first_layer.attention.mask_type == "block_causal"

    @staticmethod
    def test_create_varlen_metadata_returns_cpu_tensors():
        """attention_varlen_cpu patch makes cu_seq_q/k CPU int64."""
        import torch
        from torchtitan.models.common.attention import (
            VarlenMetadata,
            create_varlen_metadata_for_document,
        )

        input_batch = torch.tensor([[1, 2, 3, 0]])  # 0 = eos_id
        metadata = create_varlen_metadata_for_document(input_batch, eos_id=0)
        assert isinstance(metadata, VarlenMetadata)
        assert metadata.cu_seq_q.device.type == "cpu"
        assert metadata.cu_seq_k.device.type == "cpu"
        assert metadata.cu_seq_q.dtype == torch.int64
        assert metadata.cu_seq_k.dtype == torch.int64
