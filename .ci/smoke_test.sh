# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip install -r requirements.txt
pip install -r requirements_dev.txt

# Global variable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output"
REPORT_DIR="${PROJECT_ROOT}/test_reports"
INTEGRATION_REPORT_DIR="${PROJECT_ROOT}/test_reports/integration_tests"
TORCHTITAN_VERSION="v0.2.2"
TORCHTITAN_DIR="${PROJECT_ROOT}/third_party/torchtitan"
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}
SMOKE_STEPS=${SMOKE_STEPS:-1}

# Prepare environment: install packages and clone torchtitan source.
_setup_env() {
    cd "$PROJECT_ROOT"

    # Ensure torchtitan_npu is installed
    if ! python3 -c "import torchtitan_npu" 2>/dev/null; then
        python3 -m pip install -e .
    fi

    # Ensure inductor_npu_ext is installed (required by torch.compile)
    if ! python3 -c "import inductor_npu_ext" 2>/dev/null; then
        echo "Installing inductor_npu_ext..."
        if [[ ! -d "third_party/torchair" ]]; then
            mkdir -p third_party
            git clone --depth 1 \
                https://gitcode.com/Ascend/torchair.git third_party/torchair
        fi
        pip3 install -e third_party/torchair/experimental/_inductor_npu_ext/python/
    fi

    # Clone torchtitan source if not exists
    if [[ ! -d "$TORCHTITAN_DIR" ]]; then
        echo "Cloning torchtitan source..."
        mkdir -p third_party
        git clone --branch $TORCHTITAN_VERSION --depth 1 \
            https://gitcode.com/GitHub_Trending/to/torchtitan.git $TORCHTITAN_DIR
    fi
}

# Run integrated test: end-to-end training configurations.
run_torchtitan_npu_smoke() {
    echo "Running torchtitan-npu smoke test..."

    mkdir -p "$REPORT_DIR"

    local smoke_log="${REPORT_DIR}/smoke_test.log"
    local start_time=$(date +%s)
    local integration_test="${PROJECT_ROOT}/tests/smoke_tests/integration_test.py"
    echo "Verifying torchtitan..."
    python -c "import torchtitan; print('torchtitan ok')"

    echo "Done."

    cd "$PROJECT_ROOT"

    echo "Starting torchtitan-npu integration test..."

    set +e
    timeout $TIMEOUT_SECONDS bash -c "
        python "${integration_test}" "${INTEGRATION_REPORT_DIR}" --ngpu 2
    " 2>&1 | tee "$smoke_log"
    local exit_code
    exit_code=${PIPESTATUS[0]}
    set -e

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "torchtitan-npu smoke test finished in ${duration}s"
    if ! analyse_smoke_result "$smoke_log" "$exit_code"; then
        echo "torchtitan-npu smoke test failed."
        exit 1
    fi
}

analyse_smoke_result() {
    local log_file="$1"
    local exit_code="$2"

    echo "Analyzing smoke test results..."
    local has_error=false

    if [[ "$exit_code" -eq 124 ]]; then
        echo "Timeout!"
        has_error=true
    fi

    if grep -qiE "error|exception|traceback" "$log_file" 2>/dev/null; then
        has_error=true
    fi

    if grep -qiE "loss[^:]*:[^0-9]*(nan|inf)" "$log_file" 2>/dev/null \
    || grep -qiE "(nan|inf).*loss" "$log_file" 2>/dev/null; then
        echo "loss error (NAN/Inf)"
        has_error=true
    fi

    local complete_steps=$(grep -oP "step[:\s]*\K\d+" "$log_file" 2>/dev/null | tail -1)
    complete_steps=${complete_steps:-0}
    if [[ "$complete_steps" -ge "$SMOKE_STEPS" ]]; then
        echo "Completed $complete_steps steps"
    elif [[ "$complete_steps" -gt 0 ]]; then
        echo "Only finished $complete_steps/$SMOKE_STEPS steps"
    else
        echo "No output steps detected"
    fi

    if [[ "$has_error" == "true" ]] || [[ "$exit_code" -ne 0 ]]; then
        return 1
    fi

    return 0
}

# Run upstream integration tests with NPU.
run_torchtitan_smoke() {
    echo "Running torchtitan upstream integration tests..."

    mkdir -p "$REPORT_DIR"

    local smoke_log="${REPORT_DIR}/upstream_smoke.log"
    local upstream_output_dir="${REPORT_DIR}/upstream_integration_output"
    local start_time=$(date +%s)
    local saved_pythonpath="$PYTHONPATH"

    # Disable tests that use torch.compile (Autofuse takes too long to compile on CI).
    local features_py="${TORCHTITAN_DIR}/tests/integration_tests/features.py"
    for name in "1d_compile" "1d_compile_sac_op" "2d_compile" "3d_compile"; do
        if ! grep -q "\"$name\",.*disabled=True" "$features_py" 2>/dev/null; then
            sed -i "/\"$name\",/s/$/ disabled=True,/" "$features_py"
        fi
    done

    cd "$TORCHTITAN_DIR"

    rm -rf "$upstream_output_dir"
    mkdir -p "$upstream_output_dir"

    export PYTHONPATH="${TORCHTITAN_DIR}:${PROJECT_ROOT}:${PYTHONPATH}"

    # Run upstream integration tests with torchtitan_npu patches applied.
    local cmd=(
        python3 -m tests.integration_tests.run_tests
        "$upstream_output_dir"
        --test_suite features
        --test_name all
        --ngpu 2
    )

    set +e
    "${cmd[@]}" 2>&1 | tee "$smoke_log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    cd "$PROJECT_ROOT"
    export PYTHONPATH="$saved_pythonpath"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "torchtitan integration test duration: ${duration}s"

    if [[ $exit_code -eq 0 ]]; then
        echo "torchtitan integration tests passed!"
    else
        echo "torchtitan integration tests failed (exit_code=$exit_code)"
        exit $exit_code
    fi
}


_setup_env

run_torchtitan_smoke
run_torchtitan_npu_smoke
pytest -v --tb=short tests/smoke_tests

# Smoke test success sentinel, grepped by gitcode ci. Do not modify.
echo "smoke test passed."
