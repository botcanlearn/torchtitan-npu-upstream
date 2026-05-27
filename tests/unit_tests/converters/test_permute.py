# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.models.moe import MoE

from torchtitan_npu.converters.kernels.permute import (
    _npu_moe_forward_for_dsv32,
    _npu_moe_token_combine,
    _npu_moe_token_dispatch,
    PermuteKernel,
)


def test_permute_converter_applies_moe_forward_and_ep_hooks():
    original_moe_forward = MoE.forward
    original_token_dispatch = ExpertParallel._token_dispatch
    original_token_combine = ExpertParallel._token_combine

    counts = PermuteKernel.apply(None, "deepseek_v3")

    # 1 MoE.forward replacement + 1 _token_dispatch + 1 _token_combine.
    assert counts == 3
    assert MoE.forward == _npu_moe_forward_for_dsv32
    assert ExpertParallel._token_dispatch == _npu_moe_token_dispatch
    assert ExpertParallel._token_combine == _npu_moe_token_combine
    assert MoE.forward != original_moe_forward
    assert ExpertParallel._token_dispatch != original_token_dispatch
    assert ExpertParallel._token_combine != original_token_combine
