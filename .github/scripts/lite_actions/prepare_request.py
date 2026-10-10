#!/usr/bin/env python3
"""Encode workflow_dispatch inputs as an Actions artifact for an outbound-only CI agent.

A Job uploads a tiny JSON artifact; Lite Actions reads it by the run's REST API.
Never compose a shell command from test identifiers/params.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re

TEST = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
KEY = re.compile(r"[A-Z][A-Z0-9_]*\Z")


def parse_cases(value: str) -> list[dict]:
    if len(value.encode()) > 8192:
        raise ValueError('test_cases input too long')
    cases = json.loads(value)
    if not isinstance(cases, list) or not 1 <= len(cases) <= int(os.environ.get('CI_MAX_CASES','2')):
        raise ValueError('test_cases exceeds the supported case budget')
    for case in cases:
        if not isinstance(case,dict) or len(case)!=2 or 'params' not in case or (('test_id' in case)==('suite' in case)):
            raise ValueError('each request needs exactly one test_id or suite, and params')
        name=case.get('test_id',case.get('suite'))
        if not isinstance(name,str) or not TEST.fullmatch(name):
            raise ValueError('invalid registered test/suite')
        params=case['params']
        if not isinstance(params,dict) or set(params)-{'STEPS'}:
            raise ValueError('only STEPS parameter is permitted')
        if 'STEPS' in params:
            v=params['STEPS']
            if not isinstance(v,str) or not v.isdecimal() or not 1<=int(v)<=1000:
                raise ValueError('STEPS must be 1..1000')
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
