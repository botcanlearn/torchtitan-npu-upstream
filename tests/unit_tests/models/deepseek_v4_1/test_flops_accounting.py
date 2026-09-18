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


def test_ratio_one_layers_keep_their_compressed_container():
    """The ratio-1 layers of the 30-layer topology still select compressed entries.

    The 30-layer crop is ``(0, 0) + (2,) * 18 + (1,) * 10``: dropping the ratio-1
    layers from the account would lose ten layers' worth of selected-entry
    attention and the three ratio-1 index sources (20, 24, 28) that score.
    """
    config = model_registry("deepseek_v4_1_flash_30layers_16experts_vision").model

    ratio_one = [layer.attention for layer in config.layers if layer.attention.compress_ratio == 1]
    assert len(ratio_one) == 10
    assert sum(1 for attention in ratio_one if attention.indexer.mode.value != "reuse") == 3

    _, flops = config.get_nparams_and_flops(_NoParameters(), seq_len=4096)

    # 30 x window + 28 x selected + (3 full/reindex ratio-2 + 3 index-source
    # ratio-1) x indexer: the ratio-1 container is counted, so the result is
    # larger than the same account that stops at compress_ratio > 1.
    window = 30 * 6 * 64 * 512 * 2 * 128
    selected = 28 * 6 * 64 * 512 * 2 * 512
    indexer = 3 * 6 * 32 * 128 * 2048 + 3 * 6 * 32 * 128 * 4096
    assert flops == window + selected + indexer
    assert flops > window + 18 * 6 * 64 * 512 * 2 * 512 + 3 * 6 * 32 * 128 * 2048
