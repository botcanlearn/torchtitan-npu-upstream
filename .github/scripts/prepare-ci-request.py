#!/usr/bin/env python3
"""Encode workflow_dispatch inputs as an Actions artifact for an outbound-only CI agent.

A Job uploads a tiny JSON artifact; Lite Actions reads it by the run's REST API.
Never compose a shell command from test paths/params.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re

PATH = re.compile(r"tests/integration_tests/nightly_all_models_test/[a-z][a-z0-9_]*_tests\.py\Z")
TEST = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
KEY = re.compile(r"[A-Z][A-Z0-9_]*\Z")


def parse_cases(value: str) -> list[dict]:
    if len(value.encode()) > 8192:
        raise ValueError('test_cases input too long')
    cases = json.loads(value)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ValueError('test_cases must be a JSON array of 1..12 entries')
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'path', 'test_id', 'params'}:
            raise ValueError('each test needs path, test_id and params')
        if not isinstance(case['path'], str) or not PATH.fullmatch(case['path']):
            raise ValueError('invalid test module path')
        if not isinstance(case['test_id'], str) or not TEST.fullmatch(case['test_id']):
            raise ValueError('invalid registered test_id')
        params = case['params']
        if not isinstance(params, dict) or len(params) > 12 or any(
            not isinstance(k,str) or not KEY.fullmatch(k)
            or not isinstance(v,str) or len(v) > 120 or any(c in v for c in '\r\n\x00')
            for k,v in params.items()
        ):
            raise ValueError('invalid test parameters')
    return cases


def main() -> None:
    cases = parse_cases(os.environ['CI_TEST_CASES'])
    result = {'schema':1, 'run_id':int(os.environ['GITHUB_RUN_ID']),
              'attempt':int(os.environ['GITHUB_RUN_ATTEMPT']),
              'sha':os.environ['GITHUB_SHA'], 'cases':cases}
    Path('.ci-request').mkdir(exist_ok=True)
    Path('.ci-request/ci-request.json').write_text(json.dumps(result, separators=(',',':'))+'\n')
    print('Prepared CI request for', len(cases), 'test case(s).')

if __name__ == '__main__':
    main()
