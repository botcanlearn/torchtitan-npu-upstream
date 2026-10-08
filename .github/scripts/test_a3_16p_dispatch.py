"""CPU-only checks for A3 16P distributed command construction and test oracle."""
import contextlib
import io
import json
import subprocess
import importlib.util
import os
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import tempfile

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dispatch = load("ci_dispatch_16p", ROOT / "a3-16p-dispatch.py")
runner = load("ci_runner_16p", ROOT.parent.parent / "tests/integration_tests/run_16p_multinode_tests.py")


class MultinodeTests(TestCase):
    def test_topology_and_command(self):
        env = dispatch.remote_env('/mnt/share/ci_tests/example')
        for key in ('NNODES=2', 'NGPU=8', 'COMPILE_ENABLE=0',
                    'STEPS=5', 'ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
                    'NODE_IPS=192.168.0.30,192.168.0.107'):
            self.assertIn(key, env)
        cmd = dispatch.node_command('/mnt/share/ci_tests/example/node0')
        self.assertIn('run_16p_multinode_tests launch',cmd)
        self.assertIn('timeout --signal=TERM',cmd)
        self.assertIn('exit_code.txt',cmd)
        self.assertEqual(
            subprocess.run(['bash','-n','-c',cmd],capture_output=True).returncode, 0
        )
        self.assertEqual(len(dispatch.HOSTS),2)
        self.assertEqual(len(dispatch.IPS),2)

    def test_launch_and_tb_oracle(self):
        out = Path('/tmp/dsv4_16p_test')
        cmd = runner.build_launch_command(out)
        self.assertIn('deepseek_v4_flash_16p_cpt_4k_a3.sh',cmd[1])
        self.assertIn('--metrics.enable_tensorboard',cmd)
        self.assertIn('--metrics.save_tb_folder=tb_phase_0',cmd)
        with patch.object(runner,'extract_losses_from_tensorboard',return_value={1:12.,2:11.,3:10.,4:9.,5:8.}):
            runner.verify(out,5)
        with patch.object(runner,'extract_losses_from_tensorboard',return_value={1:12.,2:11.,4:9.}):
            with self.assertRaisesRegex(RuntimeError,'expected steps'):
                runner.verify(out,5)

    def test_bounded_result_protocol(self):
        full = "\n".join(f"line {i}" for i in range(31))
        with patch.object(dispatch, 'ssh', return_value=subprocess.CompletedProcess(
            [], 0, full, ''
        )):
            for statuses, verified, status in [
                ({dispatch.HOSTS[0]: 0, dispatch.HOSTS[1]: 0}, True, 'PASS'),
                ({dispatch.HOSTS[0]: 0, dispatch.HOSTS[1]: 1}, False, 'FAIL'),
            ]:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    dispatch.print_result('/mnt/share/ci_tests/test', statuses, verified)
                line = next(x for x in output.getvalue().splitlines()
                            if x.startswith('[16P_RESULT] '))
                parsed = json.loads(line.removeprefix('[16P_RESULT] '))
                self.assertEqual(parsed['status'], status)
                self.assertEqual(len(parsed['last_20_lines']), 20)
                self.assertEqual(parsed['last_20_lines'][0], 'line 11')
                self.assertEqual(parsed['last_20_lines'][-1], 'line 30')
                self.assertIn('[16P_UPLOAD_BYTES_ESTIMATE]', output.getvalue())

    def test_bad_config_rejected_before_launch(self):
        with patch.dict(os.environ,{'NODE_IPS':'192.168.0.30','HF_ASSETS_PATH':'/tmp','NGPU':'8','STEPS':'5'}):
            # The env error is surfaced before any TorchRun invocation.
            with tempfile.TemporaryDirectory() as folder:
                with patch('sys.argv',['runner','launch',folder]):
                    with self.assertRaises(SystemExit) as ctx:
                        runner.main()
                self.assertEqual(ctx.exception.code,2)


if __name__ == '__main__':
    main()
