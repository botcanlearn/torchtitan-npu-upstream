"""Fixed SHA test discovery and safe exact ID / full module suite selection."""
import pytest
from tests.integration_tests.tools.lite_actions.entrypoint import catalog,select

def test_formal_testcases_are_not_owned_by_lite_actions():
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]/'integration_tests'
    assert (root/'nightly_all_models_test/a3_8p_tests.py').is_file()
    assert (root/'nightly_all_models_test/runner.py').is_file()
    assert (root/'tools/lite_actions/entrypoint.py').is_file()
    assert not (root/'lite_actions/nightly_all_models_test').exists()

def test_discovery_returns_two_distinct_8p_cases():
    found=catalog();tests=select(found,suite='a3_8p_tests')
    assert [t.test_name for t in tests]==['dsv4_flash_a3_8p_example','dsv4_flash_a3_8p_multicase']
    assert all((x.nnodes,x.ngpu)==(1,8) for x in tests)
    assert len(select(found,test_id=tests[0].test_name))==1
    with pytest.raises(ValueError):select(found,test_id='nonexistent')
    with pytest.raises(ValueError):select(found,suite='../../escape')

def test_distributed_uses_exactly_one_cli_phase_and_tensorboard_expected_steps(monkeypatch,tmp_path):
    from tests.integration_tests.nightly_all_models_test import runner,a3_16p_tests
    test=a3_16p_tests.build_test_list()[0]
    assert test.nnodes==2
    assert '--optimizer.name' in test.override_args[0]
    assert 'torchtitan_npu.override.common.optimizer.swap_optimizer' not in test.override_args[0]
    monkeypatch.setenv('HF_ASSETS_PATH',str(tmp_path))
    monkeypatch.setenv('NODE_IPS','1.2.3.4,5.6.7.8')
    monkeypatch.setenv('NGPU','8')
    captured=[]
    def launcher(cmd,env):captured.append(cmd);return 0
    monkeypatch.setattr(runner.subprocess,'call',launcher)
    with pytest.raises(SystemExit) as exc: runner.run_distributed(test,nnodes=2,phase='launch',output_dir=tmp_path/'out')
    assert exc.value.code==0
    assert list(test.override_args[0])==captured[0][6:-1]
    assert '--training.steps' in captured[0]
    monkeypatch.setattr(__import__('tests.integration_tests.loss_compare',fromlist=['extract_losses_from_tensorboard']),
        'extract_losses_from_tensorboard',lambda *a:{int(x):1.0 for x in test.expected_steps[0]})
    runner.run_distributed(test,nnodes=2,phase='verify',output_dir=tmp_path/'out')
