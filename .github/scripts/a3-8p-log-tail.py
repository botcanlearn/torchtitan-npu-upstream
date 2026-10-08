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
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean(line):
    return ANSI.sub("", line).rstrip("\r\n")[:MAX_CHARS]


def read_tail(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as stream:
        return [clean(line) for line in collections.deque(stream, maxlen=MAX_LINES)]


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


def select_tail(root):
    stdout = read_tail(root / "run.log")
    # Runner buffers child stdout until the test finishes. On SIGTERM/timeout
    # run.log may be completely empty, while the structured rank logs survive.
    if len(stdout) >= 5:
        return stdout
    fallback = structured_tail(root)
    if fallback:
        return fallback
    return stdout or ["No training output; inspect execution-host logs"]


if __name__ == "__main__":
    for line in select_tail(Path(sys.argv[1]))[-MAX_LINES:]:
        print(line)
