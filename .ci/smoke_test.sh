#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CI smoke entrypoint. Runs only the pytest smoke suite in tests/smoke_tests.
# The integration model ST phase lives in `.ci/integration_test.sh` and runs as
# its own CI job.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SMOKE_TESTS_START="$(date +%s)"
"${PYTHON_BIN}" -m pytest -v --tb=short tests/smoke_tests
SMOKE_TESTS_END="$(date +%s)"

echo "tests.smoke_tests finished in $((SMOKE_TESTS_END - SMOKE_TESTS_START))s"
echo "smoke test passed."
