"""CPU-only behavior contracts for Nightly Muon/AdamW and suite discovery."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess

import pytest

from tests.integration_tests.tools.lite_actions.entrypoint import catalog, select


def test_discovery_selects_muon_and_adamw_cases():
    groups = catalog()
    cases = select(groups, suite="a3_8p_tests")
    assert [x.test_name for x in cases] == [
        "dsv4_flash_a3_8p_example", "dsv4_flash_a3_8p_adamw"]
    assert all(x.expected_steps == (tuple(range(1, 6)),) for x in cases)
    assert all((x.ngpu, x.nnodes) == (8, 1) for x in cases)
    assert cases[0].env_vars is None or not cases[0].env_vars.get("OPTIMIZER_OVERRIDES")
    assert cases[1].env_vars == {"OPTIMIZER_OVERRIDES": ""}
    assert "AdamW" not in cases[0].override_args[0]
    assert "AdamW" in cases[1].override_args[0]
    assert select(groups, test_id=cases[1].test_name) == [cases[1]]
    with pytest.raises(ValueError):
        select(groups, suite="../invalid")
    with pytest.raises(ValueError):
        select(groups, test_id="unknown_case")


def expanded_argv(test, tmp_path):
    """Probe existing Bash recipe's final argv without invoking TorchTitan/NPU."""
    from tests.integration_tests.nightly_all_models_test import runner
    root = Path(runner.__file__).resolve().parents[3]
    stub = tmp_path / "scripts"
    stub.mkdir(exist_ok=True)
    filename = "run_train_multinodes.sh" if test.nnodes > 1 else "run_train.sh"
    (stub / filename).write_text('printf "%s\n" "$@" > "$CASE_ARGS_FILE"\n')
    env = {**os.environ, **(test.env_vars or {}),
           "CASE_ARGS_FILE": str(tmp_path / "args.txt"),
           "NODE_IPS": ",".join(f"192.0.2.{n}" for n in range(1, test.nnodes+1)),
           "NGPU": str(test.ngpu)}
    subprocess.run(["bash", str(root / test.train_script), *test.override_args[0]],
                   cwd=tmp_path, env=env, check=True, text=True, capture_output=True)
    return (tmp_path / "args.txt").read_text().splitlines()


def last_value(argv, key):
    index = max(i for i, item in enumerate(argv) if item == key)
    return argv[index + 1]


def effective_imports(argv):
    start = max(i for i, val in enumerate(argv) if val == "--override.imports")
    result = []
    for value in argv[start+1:]:
        if value.startswith("--"):
            break
        result.append(value)
    return result


def test_muon_adamw_shell_effective_optimizer_and_npu_imports(tmp_path):
    cases = select(catalog(), suite="a3_8p_tests")
    npu = "torchtitan_npu.override.common.rms_norm.asc"
    swap = "torchtitan_npu.override.common.optimizer.swap_optimizer"
    for index, case in enumerate(cases):
        folder = tmp_path / str(index)
        folder.mkdir()
        argv = expanded_argv(case, folder)
        assert last_value(argv, "--training.steps") == "5"
        assert "--compile.no-enable" in argv
        assert npu in effective_imports(argv)
        if index == 0:
            assert last_value(argv, "--optimizer.name") == "Muon"
            assert swap in effective_imports(argv)
        else:
            assert last_value(argv, "--optimizer.name") == "AdamW"
            assert swap not in effective_imports(argv)


def test_distributed_cli_and_verify_uses_case_expected_steps(monkeypatch, tmp_path):
    from tests.integration_tests.nightly_all_models_test import runner, a3_16p_tests
    case = a3_16p_tests.build_test_list()[0]
    assert case.nnodes == 2
    assert case.env_vars == {"CONFIG":"deepseek_v4_flash_43layers_16experts",
                             "OPTIMIZER_OVERRIDES":""}
    env_probe = tmp_path / "expanded"
    env_probe.mkdir()
    argv = expanded_argv(case, env_probe)
    assert last_value(argv, "--parallelism.expert-parallel-degree") == "16"
    assert last_value(argv, "--optimizer.name") == "AdamW"
    assert "torchtitan_npu.override.common.optimizer.swap_optimizer" not in effective_imports(argv)
    assert "torchtitan_npu.override.common.rms_norm.asc" in effective_imports(argv)
    monkeypatch.setenv("HF_ASSETS_PATH", str(tmp_path))
    monkeypatch.setenv("NODE_IPS", "192.0.2.1,192.0.2.2")
    monkeypatch.setenv("NGPU", "8")
    seen = []
    def launch(cmd, env):
        seen.append(cmd)
        return 0
    monkeypatch.setattr(runner.subprocess, "call", launch)
    with pytest.raises(SystemExit) as e:
        runner.run_distributed(case, phase="launch", output_dir=tmp_path / "run")
    assert e.value.code == 0
    assert list(case.override_args[0]) == seen[0][6:-1]
    from tests.integration_tests import loss_compare
    monkeypatch.setattr(loss_compare, "extract_losses_from_tensorboard",
                        lambda *a: {x: 1.0 for x in case.expected_steps[0]})
    runner.run_distributed(case, phase="verify", output_dir=tmp_path / "run")
