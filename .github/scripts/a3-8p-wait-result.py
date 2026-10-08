#!/usr/bin/env python3
"""GitHub-hosted Actions waiter: print one matching remote result, exit with its status."""
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

REPO = "depeng1994/torchtitan-npu"
RUN_ID = int(os.environ["GITHUB_RUN_ID"])
ATTEMPT = int(os.environ["GITHUB_RUN_ATTEMPT"])
SHA = os.environ["GITHUB_SHA"]
TOKEN = os.environ["GH_TOKEN"]
EXPECTED_AUTHOR = "depeng1994"
MARKER = f"A3-8p-CI-RESULT {RUN_ID}:{ATTEMPT}"
# Query only recent comments to avoid old results/100-comment cap.
SINCE = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
URL = (f"https://api.github.com/repos/{REPO}/commits/{SHA}/comments?"
       + urllib.parse.urlencode({"per_page": 100, "since": SINCE}))
DEADLINE = time.monotonic() + int(os.environ.get("A3_8P_CI_TIMEOUT_SECONDS", "7400"))

def read_comments():
    req = urllib.request.Request(
        URL, headers={"Accept": "application/vnd.github+json",
                      "User-Agent": "a3-8p-ci-waiter",
                      "Authorization": f"Bearer {TOKEN}",
                      "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=25) as res:
        return json.load(res)

def find_result(comments):
    for item in reversed(comments):
        if item.get("user", {}).get("login") != EXPECTED_AUTHOR:
            continue
        body = item.get("body", "")
        if not body.startswith(MARKER + "\n"):
            continue
        try:
            report = json.loads(body.split("\n", 1)[1])
        except (ValueError, IndexError):
            continue
        if (report.get("run_id") == RUN_ID and report.get("attempt") == ATTEMPT
                and report.get("sha") == SHA and report.get("pipeline") == "dsv4-flash-8p"
                and report.get("status") in ("PASS", "FAIL")):
            return report
    return None

def main():
    print(f"Waiting for A3-8p-CI run={RUN_ID}, attempt={ATTEMPT}, commit={SHA[:12]}", flush=True)
    while time.monotonic() < DEADLINE:
        try:
            report = find_result(read_comments())
            if report:
                print(f"\n====== Remote pipeline: {report['status']} ======", flush=True)
                print(f"Remote exit code: {report['exit_code']}")
                print("\n====== Remote log (last 20 lines) ======")
                for line in report["last_20_lines"][-20:]:
                    print(line)
                print("====== End remote log ======", flush=True)
                return 0 if report["status"] == "PASS" and report["exit_code"] == 0 else 1
        except Exception as exc:
            print(f"GitHub read retry: {type(exc).__name__}", flush=True)
        time.sleep(12)
    print(f"ERROR: timed out waiting for internal agent for run {RUN_ID}", flush=True)
    return 124

if __name__ == "__main__":
    sys.exit(main())
