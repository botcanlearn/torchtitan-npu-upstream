"""Declarative multi-node CI cases; not a separate SSH/GitHub runner.

A5 64P is specification-only until hardware and launch adapter are validated.
"""
from dataclasses import dataclass

from tests.integration_tests import OverrideDefinitions


@dataclass(frozen=True)
class DistributedResources:
    pool: str
    nodes: int
    local_ngpu: int

    @property
    def world_size(self) -> int:
        if self.nodes < 1 or self.local_ngpu < 1:
            raise ValueError("positive node and device count required")
        return self.nodes * self.local_ngpu


@dataclass(frozen=True)
class DistributedCase:
    definition: OverrideDefinitions
    resources: DistributedResources

    def validate_parallelism(self, *, tp: int, pp: int, cp: int, dp_shard: int,
                             ep: int) -> int:
        world = self.resources.world_size
        base = tp * pp * cp * dp_shard
        if min(tp, pp, cp, dp_shard, ep) < 1 or world % base:
            raise ValueError(f"invalid parallelism for world_size={world}")
        if ep > world or world % ep:
            raise ValueError(f"EP {ep} cannot fit into world_size={world}")
        return world // base


def build_a5_pro_64p_case(hf_assets_path: str, *, steps: int = 5) -> DistributedCase:
    """64-rank A5 Pro candidate: EP32, DP-shard32, DP-replicate2.

    Defines intent; deliberately does not launch a training process.
    """
    if steps < 1 or not hf_assets_path.strip():
        raise ValueError("steps must be positive and HF assets path nonempty")
    resources = DistributedResources(pool="a5-mock", nodes=8, local_ngpu=8)
    definition = OverrideDefinitions(
        test_name="dsv4_pro_a5_64p",
        test_descr="DeepSeek-V4 Pro A5 64P distributed specification",
        ngpu=8,  # per node; NOT total world size
        train_script="examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh",
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        override_args=((),),
        env_vars={
            "MODULE": "torchtitan_npu.models.deepseek_v4",
            "CONFIG": "deepseek_v4_pro_61layers_32experts",
            "HF_ASSETS_PATH": hf_assets_path,
            "STEPS": str(steps),
            "NGPU": "8",
            "EP": "32",
            "DP_SHARD": "32",
            "GBS": "256",
            "USE_GOLDEN": "0",
        },
        expected_steps=(tuple(range(1, steps + 1)),),
        use_golden=False,
        check_loss=False,
        timeout=14400,
    )
    spec = DistributedCase(definition=definition, resources=resources)
    if spec.validate_parallelism(tp=1, pp=1, cp=1, dp_shard=32, ep=32) != 2:
        raise ValueError("expected DP replicate=2 for A5 64P")
    return spec
