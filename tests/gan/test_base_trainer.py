"""Tests for utils/gan/base_trainer.py."""

import os
import os.path as osp
import shutil
from typing import Dict, List, Optional, Tuple

import ignite.distributed as idist
import numpy as np
import pytest_check as check
import torch
from pytest_mock import MockerFixture
from torch.nn import Module
from torch.optim import Optimizer

from gan_facies.gan.base_trainer import BaseTrainerGAN, BatchType
from gan_facies.utils.configs import GlobalConfig
from tests.conftest import DataLoader32, DataLoader64, check_exists


class TrainerTest(BaseTrainerGAN):
    """Test class for BaseTrainerGAN."""
    def train_generator(self, gen: Module, g_optimizer: Optimizer,
                        disc: Module, real_batch: BatchType,
                        device: torch.device
                        ) -> Tuple[Module, Optimizer, Dict[str, torch.Tensor]]:
        """Trivial generator training."""
        return gen, g_optimizer, {'g_loss': (torch.tensor(1.), 'green', 6)}

    def train_discriminator(self, disc: Module, d_optimizer: Optimizer,
                            gen: Module, real_batch: BatchType,
                            device: torch.device
                            ) -> Tuple[Module, Optimizer,
                                       Dict[str, torch.Tensor]]:
        """Trivial discriminator training."""
        return disc, d_optimizer, {'d_loss': (torch.tensor(1.), 'red', 6)}

    def build_model_opt(self) -> Tuple[Module, Module, Optimizer, Optimizer]:

        class Generator(Module):
            """Simple Generator"""
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(128, 3, kernel_size=32)
                # Add gamma weights to test log

                class Attn:
                    """Trivial class for attention layer."""

                    def __init__(self) -> None:
                        # Set gamma parameter to test log
                        self.gamma = torch.nn.Parameter(torch.tensor(1.))

                self.attn1 = Attn()
                self.attn2 = Attn()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass."""
                x = torch.unsqueeze(x, -1)  # (B, C, 1)
                x = torch.unsqueeze(x, -1)  # (B, C, 1, 1)
                x = self.conv(x)
                return x

        gen = Generator()
        disc = torch.nn.Linear(4, 4)
        g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.001)
        d_optimizer = torch.optim.Adam(disc.parameters(), lr=0.001)

        gen = idist.auto_model(gen)
        disc = idist.auto_model(disc)
        g_optimizer = idist.auto_optim(g_optimizer)
        d_optimizer = idist.auto_optim(d_optimizer)
        return gen, disc, g_optimizer, d_optimizer

    def generate_data(self, gen: Module
                      ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate data."""
        data = np.random.rand(64, 128, 128, 3) * 255
        data = data.astype(np.uint8)
        attentions = torch.tensor([0, 1, 2, 3])
        return data, [attentions]


def build_trainers() -> Tuple[BaseTrainerGAN, BaseTrainerGAN]:
    """Return trainers with data size 32 and 64."""
    config32 = GlobalConfig().build_from_argv(
        fallback='tests/configs/data32.yaml')
    config64 = GlobalConfig().build_from_argv(
        fallback='tests/configs/data64.yaml')
    trainer32 = TrainerTest(DataLoader32(), config32)
    trainer64 = TrainerTest(DataLoader64(), config64)
    return trainer32, trainer64


# Test TrainerSAGAN

def test_init() -> None:
    """Test init method."""
    build_trainers()
    # Remove tmp folders
    if osp.exists('tests/configs/runs/tmp_test'):
        shutil.rmtree('tests/configs/runs/tmp_test')


def test_train(mocker: MockerFixture) -> None:
    """Test train method."""
    # Mock third party functions
    w_dists = {'ind1_cls_1': 0.1, 'ind1_cls_2': 0.2, 'ind2_cls_1': 0.3,
               'ind2_cls_2': 0.4, 'global': 0.2}, {'cond_acc': 0.5}
    mocker.patch('gan_facies.metrics.compute_save_indicators')
    mocker.patch('gan_facies.metrics.evaluate', return_value=w_dists)
    mocker.patch('gan_facies.utils.auxiliaries.get_delta_eta',
                 return_value=('02h02m10s', '02h05m10s'))

    # Create random datasets
    data32 = np.random.randint(0, 4, size=(5, 32, 32), dtype=np.uint8)
    data64 = np.random.randint(0, 4, size=(5, 32, 32), dtype=np.uint8)
    os.makedirs('tests/datasets', exist_ok=True)
    np.save('tests/datasets/data32.npy', data32)
    np.save('tests/datasets/data64.npy', data64)

    # Mock train_discriminator and train_generator
    def side_effect_disc(disc: Module, d_optimizer: Optimizer,
                         *args: Optional[Tuple], **kwargs: Optional[Dict]
                         ) -> Tuple[Module, Optimizer,
                                    Dict[str, torch.Tensor]]:
        # pylint: disable=unused-argument
        losses = {'d_loss': (torch.tensor(0.3), 'red', 6)}
        return disc, d_optimizer, losses

    def side_effect_gen(gen: Module, g_optimizer: Optimizer,
                        *args: Optional[Tuple], **kwargs: Optional[Dict]
                        ) -> Tuple[Module, Optimizer,
                                   Dict[str, torch.Tensor]]:
        # pylint: disable=unused-argument
        losses = {'g_loss': (torch.tensor(0.2), 'green', 6)}
        return gen, g_optimizer, losses

    trainers = build_trainers()
    for trainer in trainers:
        trainer.train_discriminator = mocker.MagicMock(  # type: ignore
            side_effect=side_effect_disc)
        trainer.train_generator = mocker.MagicMock(  # type: ignore
            side_effect=side_effect_gen)
        trainer.train()
        check_exists('res/tmp_test/models/generator_step_2.pth')
        check_exists('res/tmp_test/models/discriminator_step_2.pth')
        check_exists('res/tmp_test/samples/images_step_2.png')
        check_exists('res/tmp_test/attention/gen_attn_step_2/attn_0.npy')
        # Remove tmp folders
        shutil.rmtree('res/tmp_test')

    # Test get_log_to_dict method
    logs = trainer.get_log_to_dict(
        {
            'g_loss': (torch.tensor(0.3), 'green', 6),
            'd_loss': (torch.tensor(-0.2), 'red', 6),
            'test': (torch.tensor(0.2), 'blue', 6),
        }, avg_gammas=[0.5, 0.6, 0.7])
    expected_logs = {
        'g_loss': 0.3,
        'd_loss': -0.2,
        'test': 0.2,
        'sum_losses': 0.1,
        'abs_losses': 0.5,
        'avg_gamma1': 0.5,
        'avg_gamma2': 0.6,
        'avg_gamma3': 0.7
    }
    for key, val in expected_logs.items():
        check.is_true(np.isclose(logs[key], val), f'error for key {key}')

    if osp.exists('tests/configs/runs/tmp_test'):
        shutil.rmtree('tests/configs/runs/tmp_test')
    shutil.rmtree('tests/datasets')
