"""Tests for utils/sagan/trainer.py."""

import os
import os.path as osp
import shutil
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.configs import GlobalConfig
from utils.data.data_loader import DataLoaderMultiClass
from utils.sagan.trainer import TrainerSAGAN


class DataLoader64(DataLoaderMultiClass):
    """Data loader for unit tests (data size 64)."""

    def __init__(self) -> None:
        super().__init__()
        self.n_classes = 4

    def loader(self) -> DataLoader:
        """Return pytorch data loader."""
        return torch.utils.data.DataLoader(
            dataset=torch.randn(10, 4, 64, 64), batch_size=2,
            shuffle=True,
            num_workers=0,
        )


class DataLoader32(DataLoaderMultiClass):
    """Data loader for unit tests (data size 32)."""

    def __init__(self) -> None:
        super().__init__()
        self.n_classes = 4

    def loader(self) -> DataLoader:
        """Return pytorch data loader."""
        return torch.utils.data.DataLoader(
            dataset=torch.randn(10, 4, 32, 32), batch_size=2,
            shuffle=True,
            num_workers=0,
        )


def build_trainers() -> Tuple[TrainerSAGAN, TrainerSAGAN]:
    """Return trainers with data size 32 and 64."""
    config32 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data32.yaml')
    config64 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data64.yaml')
    trainer32 = TrainerSAGAN(DataLoader32(), config32)
    trainer64 = TrainerSAGAN(DataLoader64(), config64)
    return trainer32, trainer64


# Test TrainerSAGAN


def test_init() -> None:
    """Test init method."""
    build_trainers()
    # Remove tmp folders
    if osp.exists('configs/runs/tmp_test'):
        shutil.rmtree('configs/runs/tmp_test')


def test_train() -> None:
    """Test train method."""
    # Create random datasets
    data32 = np.random.randint(0, 4, size=(5, 32, 32), dtype=np.uint8)
    data64 = np.random.randint(0, 4, size=(5, 32, 32), dtype=np.uint8)
    os.makedirs('tests/datasets', exist_ok=True)
    np.save('tests/datasets/data32.npy', data32)
    np.save('tests/datasets/data64.npy', data64)

    trainers = build_trainers()
    for i, trainer in enumerate(trainers):
        trainer.train()
        assert osp.exists('res/tmp_test/models/generator_step_2.pth')
        assert osp.exists('res/tmp_test/models/discriminator_step_2.pth')
        assert osp.exists('res/tmp_test/samples/images_step_2.png')
        if i == 0:
            assert osp.exists('res/tmp_test/metrics/boxes_step_4.png')
            assert osp.exists('res/tmp_test/metrics/metrics_step_4.json')
        assert osp.exists('res/tmp_test/attention/gen_attn_step_2/attn_0.npy')
        # Remove tmp folders
        shutil.rmtree('res/tmp_test')

    # Test get_log_to_dict method
    logs = trainer.get_log_to_dict({'g_loss': torch.tensor(0.3),
                                    'd_loss': torch.tensor(-0.2),
                                    'test': torch.tensor(0.2)},
                                   avg_gammas=[0.5, 0.6, 0.7])
    expected_logs = {'g_loss': 0.3, 'd_loss': -0.2, 'test': 0.2,
                     'sum_losses': 0.1, 'abs_losses': 0.5,
                     'avg_gamma1': 0.5, 'avg_gamma2': 0.6,
                     'avg_gamma3': 0.7}
    for key, val in expected_logs.items():
        assert np.isclose(logs[key], val), f'error for key {key}'

    if osp.exists('configs/runs/tmp_test'):
        shutil.rmtree('configs/runs/tmp_test')
    assert osp.exists('tests/datasets/data32_ds32_co2_us4_indicators.json'), (
        'indicators not saved'
        )
    assert osp.exists('tests/datasets/data64_ds64_co1_us6_indicators.json'), (
        'indicators not saved'
        )
    shutil.rmtree('tests/datasets')
