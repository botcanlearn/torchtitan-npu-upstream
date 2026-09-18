# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 FLOPs account: window + compressed container + indexer scoring.

The parameter half of ``get_nparams_and_flops`` is the pinned torchtitan
0.3.0 helper's contract and needs a full (``ParallelDims``-backed) build, so
these tests hand the config a model with no parameters.  That cancels the
helper's per-layer full-attention estimate against the subtraction the V4.1
accounting makes and leaves exactly the attention terms the V4.1 topology
implies: the sliding window on every layer, the selected compressed entries on
every compressing layer (ratio 1 included: it pools one token per entry rather
than skipping the container), and the indexer's scoring pass only on the layers
that score -- Full and Reindex Mode, never Reuse Mode.
"""

import pytest

from torchtitan_npu.models.deepseek_v4_1 import model_registry


class _NoParameters:
    """A model stand-in: the helper reads nothing but ``named_parameters``."""

    def named_parameters(self):
        return iter(())


@pytest.mark.parametrize(
    ("flavor", "seq_len", "expected"),
    (
        # 40 x window(6*8*64*2*128) + 38 x selected(6*8*64*2*32)
        #   + (3 full + 5 reindex) x indexer(6*8*32*4096)
        # The canonical CI run's seq_len, where the two corrections nearly
        # cancel: 18 ratio-2 layers lose their over-counted 256-entry scoring
        # pass, 20 ratio-1 layers gain selected entries, and the 5 ratio-1
        # index sources start scoring their 512-entry container.
        ("deepseek_v4_1_debugmodel", 512, 31_457_280 + 7_471_104 + 5_111_808),
        ("deepseek_v4_1_debugmodel", 4096, 31_457_280 + 7_471_104 + 40_894_464),
        # 40 x window(6*64*512*2*128) + 38 x selected(6*64*512*2*512)
        #   + 3 x indexer(6*32*128*2048) + 5 x indexer(6*32*128*4096)
        ("deepseek_v4_1_flash_40layers_16experts_vision", 4096, 2_013_265_920 + 7_650_410_496 + 654_311_424),
    ),
)
def test_flops_are_window_plus_compressed_plus_indexer(flavor, seq_len, expected):
    config = model_registry(flavor).model

    nparams, flops = config.get_nparams_and_flops(_NoParameters(), seq_len=seq_len)

    assert nparams == 0
    assert flops == expected
