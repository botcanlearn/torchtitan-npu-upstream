# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared torchtitan setup for CI scripts.
# Usage: source .ci/setup_torchtitan.sh

TORCHTITAN_BRANCH="main"
TORCHTITAN_COMMIT="ac13e536c84e7f6647b14fa9375c3c8a8a2b8578"

_setup_torchtitan() {
    local target_dir="${1:-/tmp/torchtitan}"

    if [[ -d "$target_dir" ]] && git -C "$target_dir" rev-parse --verify HEAD 2>/dev/null; then
        git -C "$target_dir" fetch origin "$TORCHTITAN_BRANCH"
        git -C "$target_dir" checkout "$TORCHTITAN_COMMIT"
        return 0
    fi

    echo "Cloning torchtitan source..."
    mkdir -p "$(dirname "$target_dir")"
    git clone --branch "$TORCHTITAN_BRANCH" \
        https://gitcode.com/GitHub_Trending/to/torchtitan.git "$target_dir"
    git -C "$target_dir" checkout "$TORCHTITAN_COMMIT"
}
