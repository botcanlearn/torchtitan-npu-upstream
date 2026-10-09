"""CPU-only selected Integration Test entrypoint contract."""
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tests.integration_tests import ci_entrypoint


class CIEntryPointTests(unittest.TestCase):
    def test_registered_profiles_are_structural_and_have_real_modules(self):
        registry=json.loads(ci_entrypoint.REGISTRY.read_text())
        for name,profile in registry.items():
            with self.subTest(name=name):
                self.assertEqual(name, name.lower())
                self.assertTrue(profile['module'].startswith('tests.integration_tests.'))
                self.assertIsNotNone(importlib.util.find_spec(profile['module']))
                self.assertEqual(profile['ngpu'],8)
                self.assertIn(profile['nnodes'],(1,2,8))
        self.assertEqual(registry['dsv4_pro_a5_64p']['hf_assets_path'],'')

    def test_unknown_case_cannot_run_arbitrary_code(self):
        with patch.object(sys,'argv',['ci_entrypoint','launch','os.system','/tmp/out']), \
             patch.object(ci_entrypoint.importlib,'import_module') as loader:
            with self.assertRaises(SystemExit):
                ci_entrypoint.main()
        loader.assert_not_called()

    def test_entrypoint_delegates_to_shared_runner_without_mutating_argv(self):
        from tests.integration_tests.nightly_all_models_test import runner
        argv=['ci_entrypoint','launch','dsv4_flash_a3_16p_example','/tmp/out']
        with patch.object(sys,'argv',argv), \
             patch.object(runner,'run_distributed') as execute:
            ci_entrypoint.main()
            self.assertEqual(sys.argv,argv)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs['nnodes'],2)
        self.assertEqual(execute.call_args.kwargs['phase'],'launch')
        self.assertEqual(execute.call_args.args[0].test_name,'dsv4_flash_a3_16p_example')

if __name__=='__main__':
    unittest.main()
