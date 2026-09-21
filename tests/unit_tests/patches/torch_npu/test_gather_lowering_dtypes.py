# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The AscendC gather-family dtype restriction patch must be active."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
try:
    pytest.importorskip("torch_npu._inductor.ascendc.lowering.gather_lowering")
except Exception:  # the torch_npu inductor import chain needs an NPU driver
    pytest.skip("torch_npu ascendc lowering stack unavailable", allow_module_level=True)


def test_gather_family_guard_excludes_int64():
    from torch_npu._inductor.ascendc.lowering.common import _LoweringGuard

    import torchtitan_npu  # noqa: F401  # applies the patch at import

    aten = torch.ops.aten
    for op in (aten.gather.default, aten.embedding.default, aten.index_select.default):
        dtypes = _LoweringGuard.dtypes_support(op)
        assert dtypes is not None, f"{op} missing from the lowering whitelist"
        assert torch.int64 not in dtypes[0], f"{op} still allows int64 inputs"
        assert torch.int32 in dtypes[0], f"{op} lost int32 inputs"
