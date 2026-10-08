#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the BSD-style license in the repository root.
"""Dispatch a DeepSeek-V4 A3 16P test: 8 NPUs on each of two SSH hosts.

Usage:
    python3 .github/scripts/a3-16p-dispatch.py --preflight
    python3 .github/scripts/a3-16p-dispatch.py --run-id 12345678

No training logs are copied back: each node owns the archive, run.log, output,
exit_code.txt. The dispatcher reports only result, bounded log and byte count.
"""

from __future__ import annotations

import argparse
import json
import datetime as dt
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

HOSTS = ("a3-3-docker-relay", "a3-4-docker-relay")
IPS = ("192.168.0.30", "192.168.0.107")
ASCEND_ENV = "/mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh"
ASSETS = "/mnt/share/models/DeepSeek-V4-Flash-bf16"
TRAINER = "tests.integration_tests.run_16p_multinode_tests"
TIMEOUT = 2700


def ssh(host: str, command: str, *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
        text=True, capture_output=capture, timeout=90, check=True,
    )


def remote_env(dest: str) -> str:
    env = {
        "ASCEND_SET_ENV_PATH": ASCEND_ENV,
        "HF_ASSETS_PATH": ASSETS,
        "CKPT_SAVE_LOAD_PATH": f"{dest}/checkpoints",
        "MODULE": "torchtitan_npu.models.deepseek_v4",
        "CONFIG": "deepseek_v4_flash_43layers_16experts",
        "NODE_IPS": ",".join(IPS),
        "Network_Interface": "enp23s0f3",
        "MASTER_PORT": "6316",
        "HCCL_IF_BASE_PORT": "30160",
        "NGPU": "8",
        "NNODES": "2",
        "STEPS": "5",
        "COMPILE_ENABLE": "0",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "TORCHINDUCTOR_NPU_BACKEND": "ascendc",
        "PYTHONUNBUFFERED": "1",
    }
    return " ".join(f"{key}={shlex.quote(val)}" for key, val in env.items())


def check_remote(host: str) -> None:
    # Check actual physical NPU usage (may belong to another Docker container);
    # checking only local 'ps' is insufficient on shared NPU hosts.
    command = (
        f"source {shlex.quote(ASCEND_ENV)} >/dev/null 2>&1; "
        f"test -d {shlex.quote(ASSETS)}; "
        "ip -o -4 addr show dev enp23s0f3 | grep -q 'inet '; "
        "npu-smi info"
    )
    result = ssh(host, f"bash -lc {shlex.quote(command)}").stdout
    proc_lines = result.split("Process id in container")[-1]
    matches = [
        line for line in proc_lines.splitlines()
        if re.search(r"\|\s*[0-3]\s+[01]\s*\|\s*\d+\s*\|", line)
    ]
    if matches:
        raise RuntimeError(
            f"{host}: selected NPUs 0-7 are occupied by existing workload. "
            f"Examples: {matches[:2]}"
        )
    print(f"[PREFLIGHT] {host}: NPU 0-7 idle", flush=True)


def stage(host: str, node_dir: str, archive: Path, sha: str) -> None:
    # /mnt/share is a shared NFS mount on the second node. Each node MUST use
    # a distinct subdirectory to prevent archive/log/output clobbering.
    ssh(host, f"mkdir -p {shlex.quote(str(Path(node_dir).parent))} && mkdir {shlex.quote(node_dir)}")
    subprocess.run(
        ["scp", "-q", str(archive), f"{host}:{node_dir}/source.tar.gz"],
        timeout=180, check=True,
    )
    cmd = (
        f"cd {shlex.quote(node_dir)} && mkdir repo && "
        "tar -xzf source.tar.gz -C repo && "
        f"printf '%s\\n' {shlex.quote(sha)} > source_sha.txt && "
        "bash -n repo/examples/deepseek_v4/debug/deepseek_v4_flash_16p_cpt_4k_a3.sh && "
        "python3 -m py_compile repo/tests/integration_tests/run_16p_multinode_tests.py"
    )
    ssh(host, cmd)
    print(f"[STAGED] {host}:{node_dir} sha={sha}", flush=True)


def node_command(node_dir: str) -> str:
    return (
        f"cd {shlex.quote(node_dir)}/repo && "
        f"source {shlex.quote(ASCEND_ENV)} >/dev/null 2>&1 && "
        f"export PYTHONPATH=\"$PWD:${{PYTHONPATH:-}}\" && "
        f"export {remote_env(node_dir).replace(' ', ' ')} && "
        f"timeout --signal=TERM --kill-after=30s {TIMEOUT}s "
        f"python3 -u -m {TRAINER} launch {shlex.quote(node_dir + '/output')} "
        f">{shlex.quote(node_dir + '/run.log')} 2>&1; "
        "rc=$?; "
        f"echo \"$rc\" > {shlex.quote(node_dir + '/exit_code.txt')}; "
        "exit \"$rc\""
    )


def run_nodes(dest: str) -> dict[str, int]:
    workers = {}
    for idx, host in enumerate(HOSTS):
        workers[host] = subprocess.Popen([
            "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=25",
            host, "bash -lc " + shlex.quote(node_command(f"{dest}/node{idx}")),
        ])
    results: dict[str, int] = {}
    try:
        for host, proc in workers.items():
            results[host] = proc.wait(timeout=TIMEOUT + 120)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        for proc in workers.values():
            proc.terminate()
        raise
    print(f"[NODE_EXITS] {results}", flush=True)
    return results


def check_results(dest: str) -> None:
    node_dir = f"{dest}/node0"
    cmd = (
        f"cd {shlex.quote(node_dir)}/repo; "
        f"source {shlex.quote(ASCEND_ENV)} >/dev/null 2>&1 && "
        f"PYTHONPATH=\"$PWD:${{PYTHONPATH:-}}\" STEPS=5 "
        f"python3 -u -m {TRAINER} verify {shlex.quote(node_dir + '/output')} "
        f">{shlex.quote(node_dir + '/verify.log')} 2>&1; "
        "rc=$?; "
        f"if [ \"$rc\" != 0 ]; then cat {shlex.quote(node_dir + '/verify.log')} >> {shlex.quote(node_dir + '/run.log')}; fi; "
        f"tail -n 4 {shlex.quote(node_dir + '/verify.log')}; exit \"$rc\""
    )
    result = ssh(HOSTS[0], "bash -lc " + shlex.quote(cmd))
    print(result.stdout, flush=True)


def print_result(dest: str, statuses: dict[str, int], verification_ok: bool) -> None:
    passed = verification_ok and all(rc == 0 for rc in statuses.values())
    failed_host = next((host for host in HOSTS if statuses.get(host) != 0), HOSTS[0])
    index = HOSTS.index(failed_host)
    node_dir = f"{dest}/node{index}"
    rc = statuses.get(failed_host, 125)
    if rc == 0 and not passed:
        rc = 1  # Master TensorBoard verification error.
    log_command = (
        f"python3 {shlex.quote(node_dir + '/repo/.github/scripts/a3-8p-log-tail.py')} "
        f"{shlex.quote(node_dir)} {rc}"
    )
    try:
        tail = ssh(failed_host, log_command).stdout.splitlines()[-20:]
    except Exception as exc:
        tail = [f"Log selection failed: {type(exc).__name__}"]
    result = {
        "status": "PASS" if passed else "FAIL",
        "node_exit_codes": statuses,
        "last_20_lines": tail,
        "result_dir": dest,
    }
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    print(f"[16P_RESULT] {payload}", flush=True)
    print(f"[16P_UPLOAD_BYTES_ESTIMATE] {len(payload.encode('utf-8'))}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Check both hosts' physical NPU availability")
    parser.add_argument("--stage-only", action="store_true", help="Distribute the exact source SHA, without using NPUs")
    parser.add_argument("--run-id", help="GitHub Actions run ID, or 'manual' for a developer run")
    parser.add_argument("--run-dir", help="Reuse an existing staged /mnt/share/ci_tests/ path")
    args = parser.parse_args()
    if args.preflight:
        for host in HOSTS:
            check_remote(host)
        return 0
    if not args.run_id and not args.run_dir:
        parser.error("--run-id or --run-dir is required")
    if args.run_dir and args.stage_only:
        parser.error("--run-dir and --stage-only are mutually exclusive")
    if args.run_id and not re.fullmatch(r"\d+|manual", args.run_id):
        parser.error("--run-id must be numeric or manual")

    if args.run_dir:
        dest = args.run_dir
        if not re.fullmatch(r"/mnt/share/ci_tests/[A-Za-z0-9_-]+", dest):
            parser.error("invalid --run-dir")
    else:
        now = dt.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        dest = f"/mnt/share/ci_tests/{now}_run-{args.run_id}_16p"
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        with tempfile.TemporaryDirectory(prefix="a3_16p_ci_") as tmp:
            archive = Path(tmp) / "source.tar.gz"
            # Streaming tarball creation avoids keeping model repo in memory.
            with archive.open("wb") as out:
                tar = subprocess.Popen(["git", "archive", sha], stdout=subprocess.PIPE)
                gzip = subprocess.Popen(["gzip", "-1"], stdin=tar.stdout, stdout=out)
                assert tar.stdout is not None
                tar.stdout.close()
                if gzip.wait() or tar.wait():
                    raise RuntimeError("git archive failed")
            for idx, host in enumerate(HOSTS):
                stage(host, f"{dest}/node{idx}", archive, sha)
    if args.stage_only:
        print(f"[STAGE_ONLY] {dest}", flush=True)
        return 0
    for host in HOSTS:
        check_remote(host)
    statuses = run_nodes(dest)
    verified = False
    if all(rc == 0 for rc in statuses.values()):
        try:
            check_results(dest)
            verified = True
        except subprocess.CalledProcessError as exc:
            print(f"[16P_VERIFY_FAIL] verification rc={exc.returncode}", flush=True)
    print_result(dest, statuses, verified)
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
