# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV4 per-parameter FSDP mixed-precision policy selection on CPU.

``_dsv4_fp32_overrides`` maps exact parameter FQNs to an FP32 policy for the
SmoE hyper-connection modules, the MoE routers, the global ``hc_head``, and the
compressor APE score biases.  ``policy_overrides`` (patches/torch/distributed/
fsdp/__init__.py) attaches the mapping to every group policy built inside the
block, and ``init_dtype_attrs`` resolves each parameter by its exact FQN.  These
tests restate the selection contract independently of the implementation's
pattern tuples: every parameter under the documented families must be
overridden, and every other parameter must fall back to the policy the training
configuration builds.
"""

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig

from tests.unit_tests.models.mtp_test_utils import SEQ_LEN, VOCAB_SIZE, build_cpu_model
from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.parallelize import _dsv4_fp32_overrides

_HC_PARAMETERS = ("hc_base", "hc_fn", "hc_scale")


def _build_policy_model():
    """A small MTP model covering every FP32 family: hc pre-heads and heads,
    MoE routers, compressor/indexer APE biases, in both layer scopes."""
    config = _make_v4_config(
        dim=32,
        n_layers=3,
        vocab_size=VOCAB_SIZE,
        n_heads=4,
        head_dim=8,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=(128, 128, 4),
        window_size=SEQ_LEN,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        moe_inter_dim=16,
        num_experts=2,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=0,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=SEQ_LEN,
        compress_rope_theta=10000.0,
        original_seq_len=65536,
        num_mtp_layers=1,
    )
    return build_cpu_model(config)


def _expected_fp32_fqns(model) -> set[str]:
    """Restate the FP32 selection contract from the module tree itself.

    Selected: hc pre/post head parameters and router gate weights under
    ``layers``/``mtp_layers``, the global ``hc_head`` parameters, and the
    compressor/indexer APE biases.  Everything else stays on the default
    policy.
    """
    expected = set()
    for fqn in (name for name, _ in model.named_parameters(remove_duplicate=False)):
        parts = fqn.split(".")
        if parts[0] == "hc_head" and len(parts) == 2 and parts[1] in _HC_PARAMETERS:
            expected.add(fqn)
            continue
        if parts[0] not in ("layers", "mtp_layers") or len(parts) < 3:
            continue
        leaf = parts[2]
        if leaf in ("hc_attn_pre", "hc_ffn_pre", "hc_head") and parts[3] in _HC_PARAMETERS:
            expected.add(fqn)
        elif leaf == "moe" and fqn.endswith("moe.router.gate.weight"):
            expected.add(fqn)
        elif fqn.endswith(("attention.compressor.ape", "attention.indexer.compressor.ape")):
            expected.add(fqn)
    return expected


def test_dsv4_fp32_overrides_cover_smoe_modules_and_ape_parameters():
    model = _build_policy_model()
    training = TrainingConfig()
    overrides = _dsv4_fp32_overrides(model, training)

    expected_fqns = _expected_fp32_fqns(model)
    overridden_fqns = set(overrides)
    assert overridden_fqns == expected_fqns, (
        f"missing={sorted(expected_fqns - overridden_fqns)} "
        f"extra={sorted(overridden_fqns - expected_fqns)}"
    )
    assert expected_fqns, "fixture must contain FP32-family parameters"

    # The default policy comes from the training configuration; the overrides
    # only change the compute dtype, so they keep its reduce/output settings.
    default_reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    for fqn, fp32_policy in overrides.items():
        assert fp32_policy.param_dtype == torch.float32, fqn
        assert fp32_policy.reduce_dtype == default_reduce_dtype, fqn
        assert fp32_policy.output_dtype is None, fqn
        assert fp32_policy.cast_forward_inputs is False, fqn


def test_dsv4_fp32_overrides_match_through_checkpoint_wrappers():
    model = _build_policy_model()
    expected = _expected_fp32_fqns(model)
    model.layers["0"].attention = checkpoint_wrapper(model.layers["0"].attention)

    overrides = set(_dsv4_fp32_overrides(model, TrainingConfig()))

    wrapped = {fqn for fqn in overrides if "_checkpoint_wrapped_module" in fqn}
    assert wrapped, "checkpoint-wrapped parameters must still be selected"
    stripped = {".".join(part for part in fqn.split(".") if part != "_checkpoint_wrapped_module") for fqn in overrides}
    assert stripped == expected


def test_matches_any_fqn_pattern_through_checkpoint_wrappers():
    from torchtitan.experiments.graph_trainer.common_utils import matches_module_fqn_pattern

    def strip_checkpoint_wrapper(fqn: str) -> str:
        return ".".join(part for part in fqn.split(".") if part != "_checkpoint_wrapped_module")

    patterns = ("layers.*.hc_attn_pre", "mtp_layers.*.moe.router", "hc_head")
    assert any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("layers.0.hc_attn_pre")) for p in patterns)
    assert any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("mtp_layers.0.moe.router")) for p in patterns)
    assert any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("hc_head")) for p in patterns)
    assert any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("layers.0._checkpoint_wrapped_module.hc_attn_pre")) for p in patterns)
    assert not any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("layers.0.attention")) for p in patterns)
    assert not any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("layers.0.moe.router")) for p in patterns)
