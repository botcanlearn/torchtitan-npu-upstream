#!/usr/bin/env bash
# GitHub Run -> exact source SHA -> A3 16P dual-SSH dispatcher.
set -euo pipefail
run_id=$1
attempt=$2
sha=$3
stamp=$4
[[ "$run_id" =~ ^[0-9]+$ && "$attempt" =~ ^[0-9]+$ && "$sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$stamp" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_run-[0-9]+_attempt-[0-9]+$ ]]
work=$(mktemp -d /tmp/a3_16p_run_XXXXXX)
trap 'rm -rf "$work"' EXIT
git -C "$work" init -q
git -C "$work" remote add origin https://github.com/depeng1994/torchtitan-npu.git
git -C "$work" fetch -q --depth=1 origin "$sha"
git -C "$work" checkout -q --detach FETCH_HEAD
actual=$(git -C "$work" rev-parse HEAD)
[[ "$actual" == "$sha" ]]
cd "$work"
set +e
python3 -u .github/scripts/a3-16p-dispatch.py --run-id "$run_id" > "$work/dispatch_report.log" 2>&1
rc=$?
set -e
python3 - "$work/dispatch_report.log" "$rc" <<'PY'
import collections
import json
from pathlib import Path
import sys

path=Path(sys.argv[1])
exit_code=int(sys.argv[2])
report_lines=path.read_text(encoding="utf-8",errors="replace").splitlines()
payloads=[line.split("[16P_RESULT] ",1)[1] for line in report_lines
          if line.startswith("[16P_RESULT] ")]
details={}
if payloads:
    try:
        details=json.loads(payloads[-1])
    except ValueError:
        pass
if details:
    lines=details.get("last_20_lines",[])[:20]
    status=details.get("status","FAIL")
    root=details.get("result_dir","unknown")
    nodes=details.get("node_exit_codes",{})
else:
    lines=list(collections.deque(report_lines,maxlen=20))
    status="FAIL"
    root="unknown"
    nodes={}
if not lines:
    lines=["No dispatcher log lines; inspect node0/node1 in execution host."]
print(f"RESULT_STATUS: {status}")
print(f"RESULT_EXIT_CODE: {exit_code}")
print(f"RESULT_DIR: {root}")
print(f"NODE_EXITS: {json.dumps(nodes, separators=(',',':'))}")
print("RESULT_LAST_20_LINES_BEGIN")
for line in lines:
    print(str(line)[:180])
print("RESULT_LAST_20_LINES_END")
PY
exit "$rc"
