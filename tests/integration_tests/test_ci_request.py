"""CPU-only GitHub Actions Inputs -> bound JSON Artifact contract."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[2]/'.github/scripts/prepare-ci-request.py'
spec=importlib.util.spec_from_file_location('prepare_ci_request',SCRIPT)
prepare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
PATH='tests/integration_tests/nightly_all_models_test/a3_8p_tests.py'
CASE={'path':PATH,'test_id':'dsv4_flash_a3_8p_example','params':{'STEPS':'5'}}

class InputsArtifactTests(unittest.TestCase):
    def test_valid_multiple_cases(self):
        self.assertEqual(prepare.parse_cases(json.dumps([CASE,CASE])),[CASE,CASE])

    def test_reject_unsafe_paths_params(self):
        for value in ('../../bin/sh','tests/integration_tests/foo.py','tests/integration_tests/nightly_all_models_test/../evil.py'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                prepare.parse_cases(json.dumps([{**CASE,'path':value}]))
        with self.assertRaises(ValueError):
            prepare.parse_cases(json.dumps([{**CASE,'params':{'STEPS':'5\nexit 1'}}]))

    def test_artifact_identity_is_git_run_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            env={'CI_TEST_CASES':json.dumps([CASE]),'GITHUB_RUN_ID':'101',
                 'GITHUB_RUN_ATTEMPT':'2','GITHUB_SHA':'a'*40}
            with patch.dict(os.environ,env),patch('os.getcwd',return_value=td):
                old=os.getcwd()
                try:
                    os.chdir(td)
                    prepare.main()
                    value=json.loads(Path('.ci-request/ci-request.json').read_text())
                    self.assertEqual(value['cases'],[CASE])
                    self.assertEqual(value['run_id'],101)
                    self.assertEqual(value['attempt'],2)
                finally:
                    os.chdir(old)

if __name__=='__main__':unittest.main()
