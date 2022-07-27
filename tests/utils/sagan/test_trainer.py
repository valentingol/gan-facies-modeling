"""Tests for utils/sagan/trainer.py."""

import os.path as osp
import shutil
from typing import Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from utils.configs import GlobalConfig
from utils.sagan.trainer import TrainerSAGAN


@pytest.fixture
def data_loaders() -> Tuple[DataLoader, DataLoader]:
    """Return trainers with data size 32 and 64."""
    data_loader_32 = torch.utils.data.DataLoader(
        dataset=torch.randn(10, 4, 32, 32), batch_size=2, shuffle=True,
        num_workers=0,
    )
    data_loader_64 = torch.utils.data.DataLoader(
        dataset=torch.randn(10, 4, 64, 64), batch_size=2, shuffle=True,
        num_workers=0,
    )
    return data_loader_32, data_loader_64


def build_trainers(
    data_loaders: Tuple[DataLoader, DataLoader]
) -> Tuple[TrainerSAGAN, TrainerSAGAN]:
    """Return trainers with data size 32 and 64."""
    config32 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data32.yaml')
    config64 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data64.yaml')
    # NOTE: creates configs at configs/runs/tmp_test

    trainer32 = TrainerSAGAN(data_loaders[0], config32)
    trainer64 = TrainerSAGAN(data_loaders[1], config64)
    return trainer32, trainer64


# Test TrainerSAGAN


def test_init(data_loaders: Tuple[DataLoader, DataLoader]) -> None:
    """Test init method."""
    trainers = build_trainers(data_loaders)
    for trainer in trainers:
        assert (hasattr(trainer, 'gen') and hasattr(trainer, 'disc')
                and hasattr(trainer, 'd_optimizer')
                and hasattr(trainer, 'g_optimizer'))
    # Remove tmp folders
    if osp.exists('configs/runs/tmp_test'):
        shutil.rmtree('configs/runs/tmp_test')


def test_train(data_loaders: Tuple[DataLoader, DataLoader]) -> None:
    """Test train method."""
    trainers = build_trainers(data_loaders)
    for trainer in trainers:
        trainer.train()
        assert osp.exists('res/tmp_test/models/generator_step_2.pth')
        assert osp.exists('res/tmp_test/models/discriminator_step_2.pth')
        assert osp.exists('res/tmp_test/samples/images_step_2.png')
        assert osp.exists('res/tmp_test/attention/gen_attn_step_2/attn_0.npy')
        # Test load_pretrained_model method
        trainer.load_pretrained_model(2)
        # Remove tmp folders
        shutil.rmtree('res/tmp_test')

    # Test get_log_for_wandb method
    logs = trainer.get_log_for_wandb({'g_loss': torch.tensor(0.3),
                                      'd_loss': torch.tensor(-0.2),
                                      'test': torch.tensor(0.2)},
                                     avg_gammas=[0.5, 0.6, 0.7])
    expected_logs = {'g_loss': 0.3, 'd_loss': -0.2, 'test': 0.2,
                     'sum_losses': 0.1, 'avg_gamma1': 0.5, 'avg_gamma2': 0.6,
                     'avg_gamma3': 0.7}
    for key, val in expected_logs.items():
        assert np.isclose(logs[key], val), f'error for key {key}'

    if osp.exists('configs/runs/tmp_test'):
        shutil.rmtree('configs/runs/tmp_test')
