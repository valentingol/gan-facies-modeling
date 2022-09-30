"""Tests for utils/gan/uncond_sagan/modules.py."""
from typing import Tuple

import numpy as np
import pytest
import pytest_check as check
import torch
from pytest_mock import MockerFixture

from gan_facies.gan.uncond_sagan.modules import (UncondSADiscriminator,
                                                 UncondSAGenerator)
from gan_facies.utils.configs import GlobalConfig
from tests.conftest import AttnMock


@pytest.fixture
def gen(configs: Tuple[GlobalConfig, GlobalConfig],
        mocker: MockerFixture) -> UncondSAGenerator:
    """Return generator for tests."""
    mocker.patch('gan_facies.gan.initialization.init_weights')
    mocker.patch('gan_facies.gan.attention.SelfAttention', AttnMock)
    mocker.patch('gan_facies.gan.spectral.SpectralNorm',
                 side_effect=lambda x: x)
    _, config64 = configs
    return UncondSAGenerator(n_classes=4, model_config=config64.model)


@pytest.fixture
def disc(configs: Tuple[GlobalConfig, GlobalConfig],
         mocker: MockerFixture) -> UncondSADiscriminator:
    """Return generator for tests."""
    mocker.patch('gan_facies.gan.initialization.init_weights')
    mocker.patch('gan_facies.gan.attention.SelfAttention', AttnMock)
    mocker.patch('gan_facies.gan.spectral.SpectralNorm',
                 side_effect=lambda x: x)
    _, config64 = configs
    return UncondSADiscriminator(n_classes=4, model_config=config64.model)


def test_sa_discriminator_fwd(disc: UncondSADiscriminator) -> None:
    """Test UncondSADiscriminator.forward."""
    x = torch.rand(size=(1, 4, 64, 64), dtype=torch.float32)
    x = x / torch.sum(x, dim=1, keepdim=True)  # normalize
    preds, att_list = disc(x, with_attn=True)
    check.equal(preds.shape, (1,))
    check.equal(len(att_list), 1)
    check.equal(att_list[0].shape, (1, 4, 1024, 1024))
    preds = disc(x, with_attn=False)
    check.is_instance(preds, torch.Tensor)


def test_sa_generator_fwd(gen: UncondSAGenerator) -> None:
    """Test UncondSAGenerator.forward."""
    z = torch.rand(size=(1, 128), dtype=torch.float32)
    data, att_list = gen(z, with_attn=True)
    check.equal(data.shape, (1, 4, 64, 64))
    check.equal(len(att_list), 1)
    check.equal(att_list[0].shape, (1, 4, 1024, 1024))
    data = gen(z, with_attn=False)
    check.is_instance(data, torch.Tensor)


def test_sa_generator_generate(gen: UncondSAGenerator,
                               mocker: MockerFixture) -> None:
    """Test SAGenerator.generate."""
    mocker.patch('gan_facies.data.process.color_data_np',
                 return_value=np.random.randint(0, 256, (1, 64, 64, 3),
                                                dtype=np.uint8))
    z = torch.rand(size=(1, 128), dtype=torch.float32)
    images, attn_list = gen.generate(z, with_attn=True)
    check.equal(images.shape, (1, 64, 64, 3))
    check.is_instance(images, np.ndarray)
    check.equal(len(attn_list), 1)
    images, attn_list = gen.generate(z, with_attn=False)
    check.equal(images.shape, (1, 64, 64, 3))
    check.equal(attn_list, [])
