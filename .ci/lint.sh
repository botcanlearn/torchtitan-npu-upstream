# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/setup_torchtitan.sh"

_setup_torchtitan /tmp/torchtitan
pip install -r /tmp/torchtitan/requirements.txt
pip install -e /tmp/torchtitan

python3 -m pre_commit run --all-files
