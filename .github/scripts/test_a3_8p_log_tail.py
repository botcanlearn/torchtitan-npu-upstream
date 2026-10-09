"""Regression tests: GitHub fail reports must include actual remote log lines."""
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest
import subprocess


ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TAIL = load("remote_tail", "a3-8p-log-tail.py")


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
            tail = TAIL.select_lines(root, 1)
            self.assertEqual(len(tail), 20)
            self.assertIn("structured", tail[0])
            self.assertIn("event_14", tail[1])
            self.assertIn("event_32", tail[-1])

    def test_success_grep_first_twenty_metrics(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            lines = []
            for i in range(16):
                lines.extend([
                    f"[case] [rank{i % 8}]:[titan] 2026-10-08 22:35:41,019"
                    f" - root - INFO - step: {i + 1} loss: 12.1 "
                    f"memory: 39GiB tps: {480 + i} tflops: 42.3\n",
                    f"[case] [rank{i % 8}]:[titan] 2026-10-08 22:35:41,019"
                    f" - root - INFO - elapsed_time_per_step: {67 + i}.0s\n",
                    "unrelated noise\n",
                ])
            (root / "run.log").write_text("".join(lines))
            result = TAIL.select_lines(root, 0)
            self.assertEqual(len(result), 20)
            self.assertIn("tps: 480", result[0])
            self.assertIn("elapsed_time_per_step: 67.0s", result[1])
            self.assertIn("tps: 489", result[18])
            self.assertIn("elapsed_time_per_step: 76.0s", result[19])
            self.assertNotIn("tps: 490", "\n".join(result))
            self.assertFalse(any("unrelated" in line for line in result))
            self.assertTrue(all(len(line) <= 180 for line in result))

    def test_success_escaped_ansi_preserves_tps(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            (root / "run.log").write_text(
                "[case] [rank7]:[titan] 2026-10-08 22:35:41,019 - root"
                " - INFO - \u001b[31mstep: 5 \u001b[32mloss: 12.09528"
                " memory: 39.09GiB \u001b[34mtps: 486 \u001b[36mtflops: 42.35\n"
                "[case] [rank7]:[titan] 2026-10-08 22:35:41,019"
                " - root - INFO - \u001b[33melapsed_time_per_step: 67.366s\n"
            )
            selected = TAIL.select_lines(root, 0)
            self.assertEqual(len(selected), 2)
            self.assertIn("tps: 486", selected[0])
            self.assertIn("elapsed_time_per_step: 67.366s", selected[1])
            self.assertNotIn("\u001b", "\n".join(selected))

    def test_failure_one_line_syntax_error_routes_last_twenty(self):
        """Intentionally corrupt ONE line of a throwaway training program."""
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            program = root / "training_smoke.py"
            program.write_text(
                "for n in range(25):\n"
                "    print(f'before-failure-{n}', flush=True)\n"
                "print('finish')\n"
            )
            original = program.read_text()
            program.write_text(original.replace("print('finish')", "print('finish'"))
            self.assertEqual(
                sum(a != b for a, b in zip(original.splitlines(), program.read_text().splitlines())),
                1,
            )
            result = subprocess.run(
                [sys.executable, str(program)], capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SyntaxError", result.stderr)
            # The remote command captures merged stdout+stderr into run.log.
            (root / "run.log").write_text(
                "".join(f"prior-output-{i}\n" for i in range(25))
                + result.stdout + result.stderr
            )
            selected = TAIL.select_lines(root, result.returncode)
            self.assertEqual(len(selected), 20)
            self.assertIn("SyntaxError", "\n".join(selected))
            self.assertTrue(all(len(line) <= 180 for line in selected))
            self.assertTrue(any("prior-output" in line for line in selected))

    def test_failure_source_location_survives_fifty_cleanup_lines(self):
        """A one-line syntax fault must remain actionable after noisy teardown."""
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            program = root / "deepseek_train_smoke.py"
            program.write_text(
                "def main():\n"
                "    value = 42\n"
                "    print(value)\n"
                "main()\n"
            )
            original = program.read_text()
            program.write_text(original.replace("print(value)", "print(value"))
            self.assertEqual(
                sum(a != b for a, b in zip(
                    original.splitlines(), program.read_text().splitlines()
                )), 1
            )
            proc = subprocess.run(
                [sys.executable, str(program)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            # Before the fix, raw tail -20 consisted ONLY of cleanup records.
            (root / "run.log").write_text(
                proc.stderr + "".join(f"cleanup log {i:02d}\n" for i in range(50))
            )
            selected = TAIL.select_lines(root, proc.returncode)
            self.assertLessEqual(len(selected), 20)
            self.assertTrue(any("deepseek_train_smoke.py" in x and "line 3" in x for x in selected))
            self.assertTrue(any("print(value" in x for x in selected))
            self.assertTrue(any("^" in x for x in selected))
            self.assertTrue(any("SyntaxError:" in x for x in selected))
            self.assertIn("cleanup log 49", selected[-1])

    def test_short_error_is_not_lost_to_rank0_fallback(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            (root / "run.log").write_text(
                '  File "/tmp/bad.py", line 9\n'
                '    raise NameError("bad")\n'
                'NameError: bad\n'
            )
            path = root / "output/case/structured_logs"
            path.mkdir(parents=True)
            (path / "training.global_rank_0.x.jsonl").write_text(
                json.dumps({"step": 2, "log_type_name": "fwd_bwd_start"}) + "\n"
            )
            result = TAIL.select_lines(root, 1)
            self.assertIn("NameError: bad", result)
            self.assertTrue(any('File "/tmp/bad.py", line 9' in x for x in result))

    def test_timeout_routes_last_twenty(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            # Genuine GNU timeout exit 124; no 8-NPU training required.
            result = subprocess.run(
                ["timeout", "--signal=TERM", "0.2s", sys.executable,
                 "-c", "import time; print('starting', flush=True); time.sleep(10)"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 124)
            (root / "run.log").write_text(
                "".join(f"previous-{x}\n" for x in range(23))
                + result.stdout + "timeout: task exceeded deadline\n"
            )
            selected = TAIL.select_lines(root, 124)
            self.assertEqual(len(selected), 20)
            self.assertIn("timeout: task exceeded deadline", selected[-1])



if __name__ == "__main__":
    unittest.main()
