"""CPU-only distributed CI case validations (no A5 hardware used)."""
import unittest

from tests.integration_tests.ci_distributed_cases import (
    build_a5_pro_64p_case, DistributedCase, DistributedResources,
)


class DistributedCasesTests(unittest.TestCase):
    def test_a5_pro_64p(self):
        case=build_a5_pro_64p_case("/models/DeepSeek-V4-tokenizer")
        self.assertEqual(case.resources.world_size,64)
        self.assertEqual(case.definition.ngpu,8)
        self.assertEqual(case.definition.env_vars["EP"],"32")
        self.assertEqual(case.definition.env_vars["DP_SHARD"],"32")
        self.assertEqual(case.validate_parallelism(tp=1,pp=1,cp=1,dp_shard=32,ep=32),2)
        self.assertEqual(case.definition.expected_steps,(tuple(range(1,6)),))

    def test_invalid_topology_rejected(self):
        case=build_a5_pro_64p_case("/models/DeepSeek-V4-tokenizer")
        with self.assertRaises(ValueError):
            case.validate_parallelism(tp=3,pp=1,cp=1,dp_shard=32,ep=32)
        with self.assertRaises(ValueError):
            case.validate_parallelism(tp=1,pp=1,cp=1,dp_shard=32,ep=128)

    def test_unconfigured_assets_rejected(self):
        with self.assertRaises(ValueError):
            build_a5_pro_64p_case("",steps=5)
        with self.assertRaises(ValueError):
            build_a5_pro_64p_case("/asset",steps=0)


if __name__=="__main__":
    unittest.main()
