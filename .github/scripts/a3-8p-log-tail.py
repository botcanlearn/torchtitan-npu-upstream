#!/usr/bin/env python3
"""Select at most 20 useful log lines on the executor, never upload full logs."""
import collections
import datetime as dt
import json
from pathlib import Path
import re
import sys

MAX_LINES = 20
MAX_CHARS = 180
ANSI = re.compile(r"(?:\x1b|\\u001b)\[[0-9;]*[A-Za-z]")
FAILURE = re.compile(
    r"\b(?:SyntaxError|IndentationError|TabError|NameError|AttributeError|"
    r"TypeError|ValueError|RuntimeError|ImportError|ModuleNotFoundError|"
    r"AssertionError|OSError|FileNotFoundError|KeyError|IndexError|"
    r"MemoryError|TimeoutError|InductorError|CalledProcessError):"
    r"|(?:fatal error|error):"
)


def clean(line):
    return ANSI.sub("", line).rstrip("\r\n")[:MAX_CHARS]


def read_tail(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as stream:
        return [clean(line) for line in collections.deque(stream, maxlen=MAX_LINES)]


def read_success_metrics(path):
    """Read the FIRST 20 matching lines, in original log order."""
    if not path.is_file():
        return []
    matches = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            # The runner may render colors as literal \u001b sequences.
            line = ANSI.sub("", raw).rstrip("\r\n")
            if "tps:" not in line and "elapsed_time_per_step" not in line:
                continue
            rank = re.search(r"\[rank\d+\]", line)
            timestamp = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]\d+", line)
            payload = line.partition(" - INFO - ")[2]
            if payload and rank and timestamp:
                line = f"{rank.group()} {timestamp.group()} {payload}"
            matches.append(clean(line))
            if len(matches) == MAX_LINES:
                break
    return matches


def read_failure_lines(path):
    """Preserve a causal error traceback even if teardown logs push it off tail."""
    if not path.is_file():
        return [], False
    tail = collections.deque(maxlen=MAX_LINES)
    preceding = collections.deque(maxlen=9)
    diagnostic = []
    follow = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = clean(raw)
            tail.append(line)
            if FAILURE.search(line):
                diagnostic = [*preceding, line]
                follow = 2
            elif follow:
                diagnostic.append(line)
                follow -= 1
            preceding.append(line)
    if not diagnostic:
        return list(tail), False

    recent_tail = list(tail)
    error_lines = [line for line in diagnostic if FAILURE.search(line)]
    file_lines = [
        line for line in diagnostic
        if re.search(r'File ["\'].+?["\'], line \d+', line)
        or re.search(r'\.py:\d+', line)
    ]
    if (
        all(line in recent_tail for line in error_lines)
        and all(line in recent_tail for line in file_lines)
    ):
        return recent_tail, True

    # Show the error's File/line/source/caret plus recent teardown context.
    # Keep 10 diagnostic lines + 8 recent lines + 2 labels = 20 maximum.
    return (
        ["Failure context (from earlier in run.log):"]
        + diagnostic[-10:]
        + ["Recent log tail:"]
        + recent_tail[-8:]
    ), True


def structured_tail(root):
    # Rank 0 is sufficient for bounded diagnostics and avoids copying 8x logs.
    paths = sorted(root.glob("output/**/structured_logs/training.global_rank_0*.jsonl"))
    if not paths:
        return []
    events = collections.deque(maxlen=MAX_LINES - 1)
    with paths[-1].open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except (ValueError, TypeError):
                continue
            stamp = entry.get("time")
            try:
                time = dt.datetime.fromtimestamp(
                    float(stamp), tz=dt.timezone(dt.timedelta(hours=8))
                ).strftime("%H:%M:%S")
            except (ValueError, TypeError, OverflowError, OSError):
                time = "--:--:--"
            step = entry.get("step", "?")
            event = entry.get("log_type_name") or entry.get("event_name") or "event"
            details = ""
            if event in ("fwd_bwd_end", "step_end", "optim_end"):
                ms = entry.get("value")
                if isinstance(ms, (int, float)):
                    details = f" duration_ms={ms:.1f}"
            events.append(clean(f"[{time} CST] step={step} {event}{details}"))
    if not events:
        return []
    return ["Log source: Rank 0 structured training events"] + list(events)


def select_lines(root, exit_code):
    if exit_code == 0:
        return read_success_metrics(root / "run.log") or [
            "No tps:/elapsed_time_per_step metrics in successful run.log"
        ]
    stdout, has_error = read_failure_lines(root / "run.log")
    # Error tracebacks take priority over rank-0 training events, even when
    # only a few stdout lines were flushed before the child exited.
    if has_error or len(stdout) >= 5:
        return stdout
    fallback = structured_tail(root)
    if fallback:
        return fallback
    return stdout or ["No training output; inspect execution-host logs"]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: a3-8p-log-tail.py <run-dir> <exit-code>")
    try:
        exit_code = int(sys.argv[2])
    except ValueError as exc:
        raise SystemExit(f"Invalid exit code: {sys.argv[2]!r}") from exc
    for line in select_lines(Path(sys.argv[1]), exit_code):
        print(line)
