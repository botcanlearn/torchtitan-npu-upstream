#!/usr/bin/env bash
# Run on the authorized dispatcher; keep state/results there, full logs on 8P host.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
state="${A3_16P_CI_STATE_ROOT:-/mnt/workspace/work/multi-agent-workflow/lite-actions/a3-16p}"
mkdir -p "$state/state" "$state/logs/system"
exec >> "$state/logs/system/agent.stdout.log" 2>&1
exec flock -n "$state/state/agent.lock" python3 -u "$here/a3-16p-agent.py"
