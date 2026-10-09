#!/usr/bin/env python3
"""Single GitHub-hosted Actions result waiter for preapproved CI cases."""
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

PIPELINES = {
    "a3-smoke": ("A3-2p-CI", "npu-smi"),
    "a3-8p": ("A3-8p-CI-Example", "a3-8p"),
    "a3-16p": ("A3-16p-CI-Example", "a3-16p"),
    "a5-64p": ("A5-64p-CI", "a5-64p"),
}


def find_result(comments, marker, sha, run_id, attempt, pipeline):
    for item in reversed(comments):
        if item.get("user", {}).get("login") != "depeng1994":
            continue
        body = item.get("body", "")
        if not body.startswith(marker + "\n"):
            continue
        try:
            report = json.loads(body.split("\n", 1)[1])
        except (ValueError, IndexError):
            continue
        if (report.get("sha") == sha and report.get("run_id") == run_id
                and report.get("attempt") == attempt
                and report.get("pipeline") == pipeline
                and report.get("status") in ("PASS", "FAIL")):
            return report
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=PIPELINES, required=True)
    args = parser.parse_args()
    display, pipeline = PIPELINES[args.case]
    run_id = int(os.environ["GITHUB_RUN_ID"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    sha = os.environ["GITHUB_SHA"]
    token = os.environ["GH_TOKEN"]
    marker = f"{display}-RESULT {run_id}:{attempt}"
    since = (dt.datetime.now(dt.timezone.utc) -
             dt.timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    url = (f"https://api.github.com/repos/depeng1994/torchtitan-npu/commits/{sha}/comments?"
           + urllib.parse.urlencode({"per_page": 100, "since": since}))
    deadline = time.monotonic() + int(os.environ.get("CI_WAIT_TIMEOUT_SECONDS", "7400"))
    print(f"Waiting for {display} run={run_id}, attempt={attempt}, SHA={sha[:12]}", flush=True)
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "lite-actions-waiter/2",
                "X-GitHub-Api-Version": "2022-11-28",
            })
            with urllib.request.urlopen(req, timeout=25) as response:
                items = json.load(response)
            result = find_result(items, marker, sha, run_id, attempt, pipeline)
            if result is not None:
                print(f"RESULT {result['status']} exit_code={result['exit_code']}")
                print("====== Remote diagnostics (max 20 lines) ======")
                for line in result.get("last_20_lines", [])[-20:]:
                    print(line)
                print("====== End diagnostics ======", flush=True)
                return 0 if result["status"] == "PASS" and result["exit_code"] == 0 else 1
        except Exception as exc:
            print(f"GitHub read retry: {type(exc).__name__}", flush=True)
        time.sleep(12)
    print("ERROR: timed out waiting for internal CI", flush=True)
    return 124


if __name__ == "__main__":
    sys.exit(main())
