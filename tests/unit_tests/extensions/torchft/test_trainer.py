# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import subprocess
import sys


def test_ft_trainer_config_and_npu_lifecycle():
    program = """
from unittest.mock import Mock, patch

from torchtitan_npu.extensions.torchft import trainer as module
from torchtitan_npu.extensions import trainer as npu_trainer

config = module.FaultTolerantTrainerEx.Config()
assert config.checkpoint.enable and config.checkpoint.enable_ft_dataloader_checkpoints
assert isinstance(config.profiler, npu_trainer.CANNProfiler.Config)
for disabled in ('enable', 'enable_ft_dataloader_checkpoints'):
    setattr(config.checkpoint, disabled, False)
    with patch.object(module.TrainerEx, '__init__') as initialize:
        try:
            module.FaultTolerantTrainerEx(config)
        except ValueError as error:
            assert 'requires checkpoint' in str(error)
        else:
            raise AssertionError(f'accepted disabled checkpoint: {disabled}')
        initialize.assert_not_called()
    setattr(config.checkpoint, disabled, True)

initialize_ft = Mock()
def initialize_once(trainer, cfg):
    initialize_ft(cfg)
    trainer.model_parts = []
    trainer.gradient_accumulation_steps = 1

with patch.object(module.FaultTolerantTrainer, '__init__', initialize_once), \\
     patch.object(npu_trainer, 'set_allow_hf32') as set_hf32, \\
     patch.object(type(config.sdc), 'build') as build_sdc:
    trainer = module.FaultTolerantTrainerEx(config)
    initialize_ft.assert_called_once_with(config)
    set_hf32.assert_called_once_with(config.training.extension.allow_hf32)
    build_sdc.assert_called_once_with(
        trainer_config=config, model_parts=[], gradient_accumulation_steps=1
    )
    assert trainer._sdc is build_sdc.return_value
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=120)
