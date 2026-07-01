# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TND attention config helpers for Qwen3 models."""


def _enable_npu_varlen_attention(model_spec):
    from torchtitan_npu.models.common.npu_varlen_attention import NPUVarlenAttention

    for layer_config in model_spec.model.layers:
        layer_config.attention.inner_attention = NPUVarlenAttention.Config()
        layer_config.attention.mask_type = "block_causal"
    _patch_update_from_config()
    return model_spec


_update_from_config_patched = False


def _patch_update_from_config():
    global _update_from_config_patched
    if _update_from_config_patched:
        return
    _update_from_config_patched = True

    from torchtitan.models.qwen3.model import Qwen3Model

    from torchtitan_npu.models.common.npu_varlen_attention import NPUVarlenAttention

    _original = Qwen3Model.Config.update_from_config

    def _patched(self, *, trainer_config, **kwargs):
        try:
            _original(self, trainer_config=trainer_config, **kwargs)
        except NotImplementedError:
            # Upstream only raises NotImplementedError for CP+Varlen and
            # PP+weight-tying.  NPUVarlenAttention supports CP, so suppress
            # the error only for our own attention type.
            inner = self.layers[0].attention.inner_attention
            if not isinstance(inner, NPUVarlenAttention.Config):
                raise

    Qwen3Model.Config.update_from_config = _patched
