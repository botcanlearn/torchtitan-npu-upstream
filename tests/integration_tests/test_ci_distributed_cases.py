"""CPU-only tests for A5 64P definitions; no NPU allocation."""
import unittest
from unittest.mock import patch
from tests.integration_tests.nightly_all_models_test.a5_64p_tests import (
    build_a5_64p_test_list, validate_parallelism,
)


class A5NightlyCasesTests(unittest.TestCase):
    def test_a5_pro_64p(self):
        with patch.dict('os.environ', {'STEPS': '5', 'HF_ASSETS_PATH': '/models/a5'}):
            case = build_a5_64p_test_list()[0]
        self.assertEqual(case.ngpu, 8)
        self.assertEqual(case.env_vars['EP'], '32')
        self.assertEqual(case.env_vars['DP_SHARD'], '32')
        self.assertEqual(validate_parallelism(), 2)
        self.assertEqual(case.expected_steps, (tuple(range(1, 6)),))

    def test_invalid_topology_rejected(self):
        with self.assertRaises(ValueError):
            validate_parallelism(dp_shard=31)
        with self.assertRaises(ValueError):
            validate_parallelism(ep=128)


if __name__ == '__main__':
    unittest.main()
