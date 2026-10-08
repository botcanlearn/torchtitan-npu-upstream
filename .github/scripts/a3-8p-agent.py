#!/usr/bin/env python3
"""A3-8p-CI pull agent. Runs only a locally hardcoded pipeline."""
import argparse
import collections
import datetime as dt
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
STATE_ROOT = Path(os.environ.get("A3_8P_CI_STATE_ROOT", "/mnt/workspace/work/multi-agent-workflow/lite-actions/a3-8p"))
REPO = "depeng1994/torchtitan-npu"
OWNER = "depeng1994"
WORKFLOW = "a3-8p-ci.yml"
STATE_DIR = STATE_ROOT / "state"
LOG_DIR = STATE_ROOT / "logs"
STATE_FILE = STATE_DIR / "processed.json"
POLL_SECONDS = int(os.environ.get("A3_8P_CI_POLL_SECONDS", "60"))
MAX_BODY_BYTES = 7000
PIPELINES = {"dsv4-flash-8p": ROOT / "a3-8p-remote-pipeline.sh"}
API = f"https://api.github.com/repos/{REPO}"


class GitHub:
    def __init__(self, token):
        self.token = token
        self.total_uploaded = 0
        self.request_count = 0

    def request(self, method, path, data=None):
        url = API + path
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "a3-8p-ci-agent/1",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {self.token}",
        }
        payload = None if data is None else json.dumps(
            data, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if payload is not None:
            headers["Content-Type"] = "application/json"
            if len(payload) > MAX_BODY_BYTES:
                raise ValueError(f"POST payload {len(payload)} exceeds {MAX_BODY_BYTES} bytes")
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        # HTTP application-layer bytes only, including auth/request headers.
        parsed = urllib.parse.urlsplit(url)
        request_line = f"{method} {parsed.path}?{parsed.query} HTTP/1.1\r\n" if parsed.query else f"{method} {parsed.path} HTTP/1.1\r\n"
        uploaded = len(request_line.encode()) + len(f"Host: {parsed.netloc}\r\n".encode())
        uploaded += sum(len(f"{key}: {value}\r\n".encode()) for key, value in headers.items())
        if payload is not None:
            uploaded += len(f"Content-Length: {len(payload)}\r\n".encode()) + len(payload)
        uploaded += 2
        self.total_uploaded += uploaded
        self.request_count += 1
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}, uploaded, len(payload or b"")


def git_token():
    token = os.environ.get("A3_8P_CI_GITHUB_TOKEN")
    if not token:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError("GitHub auth unavailable; login with gh auth login")
        token = result.stdout.strip()
    if not token:
        raise RuntimeError("GitHub token empty")
    return token


def fetch_jobs(gh):
    jobs = []
    sent_total = 0
    for event in ("workflow_dispatch", "pull_request"):
        data, sent, _ = gh.request(
            "GET",
            f"/actions/workflows/{WORKFLOW}/runs?event={event}&status=in_progress&per_page=30",
        )
        sent_total += sent
        for run in data.get("workflow_runs", []):
            if not (
                run.get("name") == "A3-8p-CI"
                and run.get("event") == event
                and run.get("status") == "in_progress"
                and run.get("head_branch") in ("master", "test/a3-8p-deepseek-v4-example-e2e")
                and run.get("actor", {}).get("login") == OWNER
                and run.get("path") == ".github/workflows/a3-8p-ci.yml"
                and isinstance(run.get("id"), int)
                and re.fullmatch("[0-9a-f]{40}", run.get("head_sha", ""))
            ):
                continue
            if event == "pull_request":
                prs = run.get("pull_requests") or []
                if not any(
                    pr.get("number") == 26
                    and pr.get("base", {}).get("ref") == "master"
                    and pr.get("head", {}).get("repo", {}).get("url", "").endswith("/repos/" + REPO)
                    for pr in prs
                ):
                    continue
            jobs.append(run)
    return jobs, sent_total


def read_state():
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save_state(data):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def tail_lines(path, lines=20):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        tail = list(collections.deque(f, maxlen=lines))
    cleaned = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s.rstrip("\r\n")) for s in tail]
    return [s[:180] for s in cleaned]


# Keep human-readable task folders in Beijing time, independent of host TZ.
BEIJING_TZ = dt.timezone(dt.timedelta(hours=8))


def task_log_dir(run):
    """Sort task directories by GitHub run creation time (Beijing UTC+08)."""
    created = run.get("created_at")
    if created:
        when = dt.datetime.fromisoformat(created.replace("Z", "+00:00"))
        if when.tzinfo is None:
            raise ValueError("GitHub run created_at must contain timezone")
        when = when.astimezone(BEIJING_TZ)
    else:
        when = dt.datetime.now(BEIJING_TZ)
    stamp = when.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = int(run["id"])
    attempt = int(run.get("run_attempt", 1))
    if run_id < 0 or attempt < 1:
        raise ValueError("invalid run identity")
    return LOG_DIR / "runs" / f"{stamp}_run-{run_id}_attempt-{attempt}"



def extract_last_20(path):
    # The dispatcher only receives the short SSH result, never full training logs.
    body = path.read_text(encoding="utf-8", errors="replace")
    head = "RESULT_LAST_20_LINES_BEGIN\n"
    end = "\nRESULT_LAST_20_LINES_END"
    if head not in body or end not in body:
        return tail_lines(path)
    lines = body.split(head, 1)[1].split(end, 1)[0].splitlines()
    return [line[:180] for line in lines[-20:]]


def execute(run):
    key = f"{run['id']}:{run.get('run_attempt', 1)}"
    path = task_log_dir(run) / "pipeline.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(PIPELINES["dsv4-flash-8p"]), str(run["id"]), str(run.get("run_attempt", 1)), run["head_sha"], path.parent.name]
    with path.open("w", encoding="utf-8") as f:
        try:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=7500).returncode
        except subprocess.TimeoutExpired:
            f.write("\nERROR: pipeline exceeded 7500s\n")
            rc = 124
        except Exception as exc:
            f.write(f"\nERROR: pipeline launch failed: {type(exc).__name__}\n")
            rc = 125
    result = {
        "run_id": run["id"],
        "attempt": run.get("run_attempt", 1),
        "sha": run["head_sha"],
        "pipeline": "dsv4-flash-8p",
        "status": "PASS" if rc == 0 else "FAIL",
        "exit_code": rc,
        "last_20_lines": extract_last_20(path),
    }
    return key, result, path


def comment_body(result):
    marker = f"A3-8p-CI-RESULT {result['run_id']}:{result['attempt']}"
    return marker + "\n" + json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def publish(gh, result):
    result["upload_bytes_before_result_post"] = gh.total_uploaded
    body = comment_body(result)
    _, sent, body_bytes = gh.request(
        "POST", f"/commits/{result['sha']}/comments", {"body": body}
    )
    return sent, body_bytes


def record_metrics(result, total_bytes, request_count, post_payload_bytes, log_path):
    task_dir = Path(log_path).parent
    task_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": result["run_id"],
        "attempt": result["attempt"],
        "status": result["status"],
        "upload_bytes_application_estimate": total_bytes,
        "requests_count": request_count,
        "final_post_payload_bytes": post_payload_bytes,
        "local_full_log": str(log_path),
        "excludes": "TCP/TLS/CONNECT/proxy/retransmissions; startup/auth/other idle polls",
    }
    metrics_path = task_dir / "upload-metrics.json"
    tmp_path = metrics_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(metrics_path)
    logging.info("upload metrics run=%s bytes=%d POST-body=%d requests=%d",
                 result["run_id"], total_bytes, post_payload_bytes, request_count)


def run_once(gh, state):
    jobs, discovery_bytes = fetch_jobs(gh)
    for run in jobs:
        key = f"{run['id']}:{run.get('run_attempt', 1)}"
        existing = state.get(key, {})
        if existing.get("state") == "reported":
            continue
        if existing.get("state") == "pending":
            result = existing["result"]
            path = Path(existing["log_path"])
            existing["upload_bytes"] += discovery_bytes
            existing["requests_count"] += 1
            state[key] = existing
            save_state(state)
        elif existing.get("state") == "running":
            logging.error("previous execution interrupted for run=%s, requires manual recovery", run["id"])
            continue
        else:
            state[key] = {"state": "running"}
            save_state(state)
            _, result, path = execute(run)
            existing = {
                "state": "pending", "result": result, "log_path": str(path),
                "upload_bytes": discovery_bytes, "requests_count": 1
            }
            state[key] = existing
            save_state(state)
        before = gh.total_uploaded
        try:
            _, post_body = publish(gh, result)
        except (urllib.error.URLError, ValueError) as exc:
            existing["upload_bytes"] += gh.total_uploaded - before
            existing["requests_count"] += 1
            state[key] = existing
            save_state(state)
            logging.warning("report pending run=%s: %s: %s", run["id"], type(exc).__name__, str(exc)[:240])
            continue
        existing["upload_bytes"] += gh.total_uploaded - before
        existing["requests_count"] += 1
        record_metrics(result, existing["upload_bytes"], existing["requests_count"], post_body, path)
        state[key] = {"state": "reported", "status": result["status"]}
        save_state(state)
        logging.info("reported run=%s %s", run["id"], result["status"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Do not use for an expensive 8P workflow")
    p.add_argument("--once", action="store_true", help="Poll once then exit")
    args = p.parse_args()
    system_log_dir = LOG_DIR / "system"
    system_log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[RotatingFileHandler(system_log_dir / "agent.log",
                                                      maxBytes=5 * 1024 * 1024,
                                                      backupCount=3),
                                  logging.StreamHandler()])
    if args.smoke:
        logging.info("8P smoke test intentionally disabled")
        return
    gh = GitHub(git_token())
    logging.info("A3-8p-CI agent started interval=%ss (fixed pipeline only)", POLL_SECONDS)
    while True:
        try:
            run_once(gh, read_state())
        except Exception as exc:
            logging.exception("poll failed: %s", type(exc).__name__)
        if args.once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
