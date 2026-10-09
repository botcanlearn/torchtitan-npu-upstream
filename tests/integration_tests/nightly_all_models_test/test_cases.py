"""No-hardware contract tests for Nightly All Models case definitions."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.nightly_all_models_test.a3_8p_tests import build_a3_8p_test_list
from tests.integration_tests.nightly_all_models_test.a3_16p_tests import build_a3_16p_test_list
from tests.integration_tests.nightly_all_models_test.a5_64p_tests import build_a5_64p_test_list
from tests.integration_tests.nightly_all_models_test import runner


class NightlyCaseTests(unittest.TestCase):
    def test_same_definition_style_per_action(self):
        cases=((build_a3_8p_test_list,8,'8p'),
               (build_a3_16p_test_list,16,'16p'),
               (build_a5_64p_test_list,64,'64p'))
        with patch.dict(os.environ,{'STEPS':'3','HF_ASSETS_PATH':'/assets','COMPILE_ENABLE':'0'}):
            for factory,world,suffix in cases:
                with self.subTest(world_size=world):
                    case_list=factory()
                    self.assertEqual(len(case_list),1)
                    case=case_list[0]
                    self.assertIsInstance(case,OverrideDefinitions)
                    self.assertEqual(case.ngpu,8)
                    self.assertIn(suffix,case.train_script)
                    self.assertEqual(case.expected_steps,(tuple(range(1,4)),))
                    self.assertFalse(case.use_golden)
                    self.assertFalse(case.check_loss)

    def test_steps_invalid(self):
        for value in ('0','-3','wrong'):
            with patch.dict(os.environ, {'STEPS':value}):
                with self.assertRaises(ValueError):
                    runner.required_steps()

    def test_distributed_validate_prevents_partial_topology(self):
        case=build_a3_16p_test_list()[0]
        with patch.dict(os.environ, {'NODE_IPS':'one', 'NGPU':'8', 'HF_ASSETS_PATH':'/tmp'}), \
             patch('tests.integration_tests.nightly_all_models_test.runner.validate_assets'), \
             patch('sys.argv',['case','launch','/tmp/result']):
            with self.assertRaises(SystemExit):
                runner.run_distributed(case,nnodes=2)


if __name__=='__main__':
    unittest.main()
