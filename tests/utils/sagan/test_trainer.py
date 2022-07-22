"""Tests for utils/sagan/trainer.py."""

import os
import os.path as osp
import shutil
from typing import Tuple

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


def build_trainers(data_loaders) -> Tuple[TrainerSAGAN, TrainerSAGAN]:
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
        assert osp.exists('res/tmp_test/attention/gen_attn0_step_2.npy')
        # Remove tmp folders
        shutil.rmtree('res/tmp_test')

    if osp.exists('configs/runs/tmp_test'):
        shutil.rmtree('configs/runs/tmp_test')


def test_load_pretrained_model(
        data_loaders: Tuple[DataLoader, DataLoader]) -> None:
    """Test load_pretrained_model method."""
    trainers = build_trainers(data_loaders)
    trainer = trainers[1]
    trainer.model_save_path = 'tests/unittest_models'
    os.makedirs('res/tmp_test/models', exist_ok=True)

    trainer.load_pretrained_model(2)

    # Remove tmp folders
    shutil.rmtree('res/tmp_test')
    if osp.exists('configs/runs/tmp_test'):
        shutil.rmtree('configs/runs/tmp_test')
