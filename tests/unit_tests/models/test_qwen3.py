# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# model_registry
# ---------------------------------------------------------------------------


def test_model_registry_returns_spec_with_npu_overrides():
    from torchtitan_npu.models.qwen3 import model_registry
    from torchtitan_npu.models.qwen3.parallelize import parallelize_qwen3

    spec = model_registry("30B-A3B")

    assert spec.parallelize_fn is parallelize_qwen3


def test_model_registry_preserves_upstream_fields():
    from torchtitan_npu.models.qwen3 import model_registry

    spec = model_registry("30B-A3B")

    assert spec.name is not None
    assert spec.flavor == "30B-A3B"
    assert spec.model is not None


def test_model_registry_invalid_flavor_raises():
    from torchtitan_npu.models.qwen3 import model_registry

    with pytest.raises(KeyError):
        model_registry("nonexistent-flavor")


# ---------------------------------------------------------------------------
# parallelize
# ---------------------------------------------------------------------------


def test_parallelize_raises_on_indivisible_n_heads():
    from torchtitan_npu.models.qwen3.parallelize import parallelize_qwen3

    mock_model = MagicMock()
    mock_model.layers = {0: SimpleNamespace(attention=SimpleNamespace(n_heads=7))}

    mock_parallel_dims = MagicMock()
    mock_parallel_dims.cp_enabled = True
    mock_parallel_dims.cp = 4
    mock_parallel_dims.tp_enabled = False

    with pytest.raises(ValueError, match="n_heads=7 must be divisible"):
        parallelize_qwen3(
            mock_model,
            parallel_dims=mock_parallel_dims,
            training=MagicMock(),
            model_converters=MagicMock(),
            parallelism=MagicMock(),
            compile_config=MagicMock(),
            ac_config=MagicMock(),
            dump_folder="/tmp/test",
        )


def _call_parallelize_qwen3_with_mock_upstream(
    titan_qwen3_parallelize,
    parallelize_qwen3,
    mock_model,
    mock_parallel_dims,
):
    """Call parallelize_qwen3 with upstream patched, return (result, mock_upstream)."""
    upstream_return = MagicMock()
    with patch.object(
        titan_qwen3_parallelize,
        "parallelize_qwen3",
        return_value=upstream_return,
    ) as mock_upstream:
        result = parallelize_qwen3(
            mock_model,
            parallel_dims=mock_parallel_dims,
            training=MagicMock(),
            model_converters=MagicMock(),
            parallelism=MagicMock(),
            compile_config=MagicMock(),
            ac_config=MagicMock(),
            dump_folder="/tmp/test",
        )
    return result, mock_upstream


def test_parallelize_without_cp_calls_upstream_directly():
    """When cp_enabled=False, parallelize_qwen3 should delegate to upstream
    without any monkey-patching on apply_cp_to_attention_module.
    """
    import torchtitan.models.qwen3.parallelize as titan_qwen3_parallelize

    from torchtitan_npu.models.qwen3.parallelize import parallelize_qwen3

    mock_model = MagicMock()
    mock_parallel_dims = MagicMock()
    mock_parallel_dims.cp_enabled = False

    result, mock_upstream = _call_parallelize_qwen3_with_mock_upstream(
        titan_qwen3_parallelize,
        parallelize_qwen3,
        mock_model,
        mock_parallel_dims,
    )

    assert result is mock_upstream.return_value
    mock_upstream.assert_called_once_with(
        mock_model,
        parallel_dims=mock_parallel_dims,
        training=mock_upstream.call_args[1]["training"],
        model_converters=mock_upstream.call_args[1]["model_converters"],
        parallelism=mock_upstream.call_args[1]["parallelism"],
        compile_config=mock_upstream.call_args[1]["compile_config"],
        ac_config=mock_upstream.call_args[1]["ac_config"],
        dump_folder="/tmp/test",
    )
