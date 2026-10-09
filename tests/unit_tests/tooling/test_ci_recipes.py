"""Offline contracts for shared training recipes; does not import torch or touch NPUs."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
FLASH = ROOT / 'examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh'
PRO = ROOT / 'examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh'


def recipe_argv(script: Path, env: dict[str, str]) -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td) / 'scripts'
        folder.mkdir()
        (folder / 'run_train_multinodes.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
        merged = {**os.environ, **env}
        response = subprocess.run(['bash',str(script)],cwd=td,env=merged,
                                  check=True,text=True,capture_output=True)
        return response.stdout.splitlines()


def read_arg(argv: list[str], name: str) -> str:
    return argv[argv.index(name)+1]


def test_flash_16p_eager_adamw_smoke_matches_previous_effective_flags():
    flags = recipe_argv(FLASH, {
        'NODE_IPS':'10.0.0.1,10.0.0.2','NGPU':'8',
        'EP':'16','DP_SHARD':'16','GBS':'128','STEPS':'5',
        'CONFIG':'deepseek_v4_flash_43layers_16experts',
        'OPTIMIZER_NAME':'AdamW','OPTIMIZER_OVERRIDES':'',
        'COMPILE_ENABLE':'0','CHECKPOINT_ENABLE':'0','FORCE_LOAD_BALANCE':'1',
        'COMM_INIT_TIMEOUT_SECONDS':'600',
    })
    assert read_arg(flags,'--parallelism.expert-parallel-degree') == '16'
    assert read_arg(flags,'--parallelism.data-parallel-shard-degree') == '16'
    assert read_arg(flags,'--parallelism.data-parallel-replicate-degree') == '1'
    assert read_arg(flags,'--training.global-batch-size') == '128'
    assert read_arg(flags,'--training.steps') == '5'
    assert read_arg(flags,'--optimizer.name') == 'AdamW'
    assert '--compile.no-enable' in flags
    assert '--checkpoint.no-enable' in flags
    assert '--debug.moe-force-load-balance' in flags
    assert read_arg(flags,'--comm.init-timeout-seconds') == '600'
    assert not any(x.startswith('--optimizer.muon') for x in flags)
    assert 'torchtitan_npu.override.common.optimizer.swap_optimizer' not in flags
    assert flags.count('--training.steps') == 1


def test_existing_flash_defaults_are_preserved():
    flags = recipe_argv(FLASH, {'NODE_IPS':','.join('10.0.0.'+str(i) for i in range(1,9)),
                                'NGPU':'16'})
    assert read_arg(flags,'--parallelism.expert-parallel-degree') == '128'
    assert read_arg(flags,'--parallelism.data-parallel-replicate-degree') == '1'
    assert read_arg(flags,'--training.global-batch-size') == '1024'
    assert read_arg(flags,'--training.steps') == '100'
    assert read_arg(flags,'--optimizer.name') == 'Muon'
    assert '--compile.enable' in flags and '--checkpoint.enable' in flags
    assert 'torchtitan_npu.override.common.optimizer.swap_optimizer' in flags


def test_existing_a5_pro_recipe_scales_32p_to_64p_without_copy():
    for nodes, replicate in ((4,'1'),(8,'2')):
        flags = recipe_argv(PRO, {'NODE_IPS':','.join('10.0.0.'+str(i) for i in range(1,nodes+1)),
                                  'NGPU':'8','STEPS':'5'})
        assert read_arg(flags,'--parallelism.expert-parallel-degree') == '32'
        assert read_arg(flags,'--parallelism.data-parallel-shard-degree') == '32'
        assert read_arg(flags,'--parallelism.data-parallel-replicate-degree') == replicate
        assert read_arg(flags,'--training.steps') == '5'
        assert '--extension.quantization.enable-quantized-training' in flags
        assert read_arg(flags,'--extension.quantization.recipe') == 'all_block_fp8'
        assert read_arg(flags,'--optimizer.name') == 'Muon'
        assert '--checkpoint.initial-load-path' in flags
