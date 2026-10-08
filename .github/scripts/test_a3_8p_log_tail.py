"""Regression tests: GitHub fail reports must include actual remote log lines."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess


ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TAIL = load("remote_tail", "a3-8p-log-tail.py")
AGENT = load("remote_agent", "a3-8p-agent.py")


class LogTailTests(unittest.TestCase):
    def test_empty_runner_log_falls_back_to_rank0_events(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            (root / "run.log").touch()
            dest = root / "output/case/test_run/structured_logs"
            dest.mkdir(parents=True)
            rows = [
                {"time": 1791465583, "step": 2, "log_type_name": f"event_{x}"}
                for x in range(33)
            ]
            (dest / "training.global_rank_0.test.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n"
            )
            tail = TAIL.select_tail(root)
            self.assertEqual(len(tail), 20)
            self.assertIn("structured", tail[0])
            self.assertIn("event_14", tail[1])
            self.assertIn("event_32", tail[-1])

    def test_success_stdout_uses_last_twenty(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            (root / "run.log").write_text(
                "".join(f"stdout-{x}\n" for x in range(31))
            )
            tail = TAIL.select_tail(root)
            self.assertEqual(tail[0], "stdout-11")
            self.assertEqual(tail[-1], "stdout-30")
            self.assertEqual(len(tail), 20)

    def test_dispatcher_pins_pipeline_script_during_execution(self):
        with tempfile.TemporaryDirectory() as dirname:
            base = Path(dirname)
            source = base / "live-pipeline.sh"
            source.write_text("echo original\n")
            captured = {}

            def fake_subprocess_run(cmd, **kwargs):
                pinned = Path(cmd[1])
                captured["cmd"] = cmd
                self.assertNotEqual(pinned, source)
                self.assertEqual(pinned.read_text(), "echo original\n")
                source.write_text("echo modified\n")
                self.assertEqual(pinned.read_text(), "echo original\n")
                return subprocess.CompletedProcess(cmd, 0)

            run = {
                "id": 42,
                "run_attempt": 1,
                "head_sha": "e" * 40,
                "created_at": "2026-10-08T13:07:51Z",
            }
            with patch.object(AGENT, "LOG_DIR", base / "logs"), \
                 patch.dict(AGENT.PIPELINES, {"dsv4-flash-8p": source}), \
                 patch.object(AGENT.subprocess, "run", side_effect=fake_subprocess_run):
                _, result, logfile = AGENT.execute(run)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(
                (logfile.parent / "remote-pipeline.sh").read_text(),
                "echo original\n",
            )

    def test_dispatcher_never_includes_end_marker(self):
        with tempfile.TemporaryDirectory() as dirname:
            p = Path(dirname) / "pipeline.log"
            p.write_text(
                "RESULT_LAST_20_LINES_BEGIN\n"
                "real error 1\nreal error 2\n"
                "RESULT_LAST_20_LINES_END\n"
            )
            self.assertEqual(
                AGENT.extract_last_20(p), ["real error 1", "real error 2"]
            )
            p.write_text(
                "RESULT_LAST_20_LINES_BEGIN\nRESULT_LAST_20_LINES_END\n"
            )
            self.assertEqual(
                AGENT.extract_last_20(p), ["No remote log lines returned"]
            )


if __name__ == "__main__":
    unittest.main()
