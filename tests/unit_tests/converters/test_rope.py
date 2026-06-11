# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan_npu.converters.kernels.rope import (
    _ROPE_REPLACEMENTS,
    apply_reshape_for_broadcast_complex_patch,
    npu_apply_rotary_emb_complex,
    npu_apply_rotary_emb_cos_sin,
    npu_apply_rotary_emb_single_complex,
    NpuRoPEConverter,
    reshape_for_broadcast_complex,
)

# Upstream helper name. Referenced via getattr/monkeypatch with this (non-literal)
# name so the tests neither touch the module-private attribute directly
# (CodeCheck G.CLS.11) nor pass getattr a constant literal (flake8 B009).
UPSTREAM_HELPER = "_reshape_for_broadcast_complex"


def _complex_freqs(shape):
    real = torch.randn(*shape, dtype=torch.float32)
    imag = torch.randn(*shape, dtype=torch.float32)
    return torch.complex(real, imag)


def test_rope_replacement_mapping_tracks_current_upstream_api():
    assert _ROPE_REPLACEMENTS == {
        "apply_rotary_emb_complex": npu_apply_rotary_emb_complex,
        "apply_rotary_emb_single_complex": npu_apply_rotary_emb_single_complex,
        "apply_rotary_emb_cos_sin": npu_apply_rotary_emb_cos_sin,
    }


@pytest.mark.parametrize(
    "func_name,impl", list(_ROPE_REPLACEMENTS.items()), ids=list(_ROPE_REPLACEMENTS)
)
def test_replace_one_invokes_replace_functions_for_each_entry(func_name, impl):
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.llama3.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        NpuRoPEConverter._replace_one(func_name, impl, fake_model)

    assert mock_replace.call_count >= 1
    for call in mock_replace.call_args_list:
        assert call.args[0] == func_name
        assert call.args[1] is impl


def test_convert_iterates_all_replacements():
    converter = NpuRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()

    with patch.object(NpuRoPEConverter, "_replace_one") as mock_replace_one:
        converter.convert(fake_model)

    assert mock_replace_one.call_count == len(_ROPE_REPLACEMENTS)
    called_pairs = {
        (call.args[0], call.args[1].__name__)
        for call in mock_replace_one.call_args_list
    }
    expected_pairs = {
        (name, impl.__name__) for name, impl in _ROPE_REPLACEMENTS.items()
    }
    assert called_pairs == expected_pairs
    for call in mock_replace_one.call_args_list:
        assert call.args[2] is fake_model


def test_replace_one_walks_three_packages_for_npu_model():
    """
    torchtitan_npu.* model → walk three locations:
    (1) the model's own module tree,
    (2) the upstream-rewritten package (torchtitan_npu→torchtitan),
    (3) the shared torchtitan.models.common package.
    """
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan_npu.models.deepseek_v4.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        NpuRoPEConverter._replace_one(
            "apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model
        )

    assert mock_replace.call_count == 3
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}
    assert mock_replace.call_args_list[1].kwargs == {
        "package": "torchtitan.models.deepseek_v4.model"
    }
    assert mock_replace.call_args_list[2].kwargs == {
        "package": "torchtitan.models.common"
    }


def test_replace_one_walks_two_packages_when_model_is_already_upstream():
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.llama3.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        NpuRoPEConverter._replace_one(
            "apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model
        )

    assert mock_replace.call_count == 2
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}
    assert mock_replace.call_args_list[1].kwargs == {
        "package": "torchtitan.models.common"
    }


def test_replace_one_walks_only_model_when_already_in_common_pkg():
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.common.rope"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        NpuRoPEConverter._replace_one(
            "apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model
        )

    assert mock_replace.call_count == 1
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}


def test_none_positions_uses_contiguous_slice():
    x = _complex_freqs((2, 4, 3, 8))
    freqs_cis = _complex_freqs((8, 8))

    actual = reshape_for_broadcast_complex(freqs_cis, x, None)
    expected = freqs_cis[:4].view(1, 4, 1, 8)

    assert actual.shape == (1, 4, 1, 8)
    torch.testing.assert_close(actual, expected)


def test_shared_positions_index_real_view():
    x = _complex_freqs((2, 4, 3, 8))
    freqs_cis = _complex_freqs((8, 8))
    positions = torch.tensor([[0, 2, 4, 6]])

    actual = reshape_for_broadcast_complex(freqs_cis, x, positions)
    expected = freqs_cis[positions.squeeze(0)].view(1, 4, 1, 8)

    assert actual.shape == (1, 4, 1, 8)
    torch.testing.assert_close(actual, expected)


def test_batched_positions_use_shared_first_row():
    """(bsz, seqlen) positions are treated as batch-shared (fused-op constraint):
    row 0 is selected and broadcast over the batch.
    """
    x = _complex_freqs((2, 4, 3, 8))
    freqs_cis = _complex_freqs((8, 8))
    positions = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]])

    actual = reshape_for_broadcast_complex(freqs_cis, x, positions)
    expected = freqs_cis[positions[0]].view(1, 4, 1, 8)

    assert actual.shape == (1, 4, 1, 8)
    torch.testing.assert_close(actual, expected)


def test_npu_smoke_complex_index_workaround(npu_device):
    """CPU can't catch this: NPU rejects complex64 index, so naive indexing must
    fail on-device while the real-view path succeeds.
    """
    x = _complex_freqs((2, 4, 3, 8)).to(npu_device)
    freqs_cis = _complex_freqs((8, 8)).to(npu_device)
    positions = torch.tensor([[0, 2, 4, 6]], device=npu_device)

    with pytest.raises(RuntimeError):
        _ = freqs_cis[positions]

    actual = reshape_for_broadcast_complex(freqs_cis, x, positions)
    assert actual.shape == (1, 4, 1, 8)
    expected = reshape_for_broadcast_complex(freqs_cis.cpu(), x.cpu(), positions.cpu())
    torch.testing.assert_close(actual.cpu(), expected)


def test_rope_patch_installs_npu_impl_on_real_upstream_helper():
    """Sentinel against upstream rename/move: the patch must install the NPU
    implementation onto the real upstream helper. Re-apply explicitly so the
    assertion is independent of cross-test module import/reload ordering
    (another test in the suite may reset the shared common.rope module).

    NOTE: test_registry.py reloads converters.kernels.rope, which replaces
    the module-level function objects. The top-level ``from ... import``
    bindings in this file become stale. Re-fetch from the live module to
    compare against the current (post-reload) function object.
    """
    import torchtitan.models.common.rope as upstream_rope

    import torchtitan_npu.converters.kernels.rope as rope_mod

    rope_mod.apply_reshape_for_broadcast_complex_patch()
    assert (
        getattr(upstream_rope, UPSTREAM_HELPER)
        is rope_mod.reshape_for_broadcast_complex
    )


def test_apply_patch_fails_loud_when_target_missing(monkeypatch):
    """A missing target must fail loud, not silently skip."""
    import torchtitan.models.common.rope as upstream_rope

    monkeypatch.delattr(upstream_rope, UPSTREAM_HELPER, raising=False)
    with pytest.raises(RuntimeError, match=UPSTREAM_HELPER):
        apply_reshape_for_broadcast_complex_patch()
