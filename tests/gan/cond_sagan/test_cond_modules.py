"""Tests for utils/gan/cond_sagan/module.py."""

from typing import Tuple

import numpy as np
import pytest
import pytest_check as check
import torch
from pytest_mock import MockerFixture

from gan_facies.gan.cond_sagan.modules import CondSAGenerator
from gan_facies.utils.configs import GlobalConfig
from tests.conftest import AttnMock


@pytest.fixture
def gen(configs: Tuple[GlobalConfig, GlobalConfig],
        mocker: MockerFixture) -> CondSAGenerator:
    """Return generator for tests."""
    mocker.patch('gan_facies.gan.initialization.init_weights')
    mocker.patch('gan_facies.gan.attention.SelfAttention', AttnMock)
    mocker.patch('gan_facies.gan.spectral.SpectralNorm',
                 side_effect=lambda x: x)
    config32, _ = configs
    return CondSAGenerator(n_classes=4, model_config=config32.model)


def test_sa_generator_fwd(gen: CondSAGenerator) -> None:
    """Test CondSAGenerator.forward."""
    pixel_maps = torch.randint(0, 2, size=(5, 4, 32, 32), dtype=torch.float32)
    z = torch.rand(size=(5, 128), dtype=torch.float32)
    data, att_list = gen(z, pixel_maps, with_attn=True)
    check.equal(data.shape, (5, 4, 32, 32))
    check.equal(len(att_list), 3)
    check.equal(att_list[0].shape, (5, 1, 16, 16))
    check.equal(att_list[1].shape, (5, 1, 64, 64))
    check.equal(att_list[2].shape, (5, 1, 256, 256))
    data = gen(z, pixel_maps, with_attn=False)
    check.is_instance(data, torch.Tensor)


def test_sa_generator_generate(gen: CondSAGenerator,
                               mocker: MockerFixture) -> None:
    """Test SAGenerator.generate."""
    mocker.patch('gan_facies.data.process.color_data_np',
                 return_value=np.random.randint(0, 256, (5, 32, 32, 3),
                                                dtype=np.uint8))
    pixel_maps = torch.randint(0, 2, size=(5, 4, 32, 32), dtype=torch.float32)
    z = torch.rand(size=(5, 128), dtype=torch.float32)
    images, attn_list = gen.generate(z, pixel_maps, with_attn=True)
    check.is_instance(images, np.ndarray)
    check.equal(images.shape, (5, 32, 32, 3))
    check.equal(len(attn_list), 3)
    images, attn_list = gen.generate(z, pixel_maps, with_attn=False)
    check.is_instance(images, np.ndarray)
    check.equal(attn_list, [])


def test_sa_generator_proba_map(gen: CondSAGenerator,
                                mocker: MockerFixture) -> None:
    """Test SAGenerator.proba_map."""
    mocker.patch('gan_facies.data.process.continuous_color_data_np',
                 return_value=np.random.randint(0, 256, (32, 32, 3),
                                                dtype=np.uint8))
    pixel_map = torch.randint(0, 2, size=(4, 32, 32), dtype=torch.float32)
    z = torch.rand(size=(5, 128), dtype=torch.float32)
    # Case batch_size = None
    proba_mean, proba_std, _ = gen.proba_map(z, pixel_map, batch_size=None)
    check.is_instance(proba_mean, np.ndarray)
    check.is_true(proba_mean.shape == (32, 32, 4))
    check.greater_equal(proba_mean.min(), 0.0)
    check.is_true(np.allclose(np.sum(proba_mean, axis=-1), 1.0))
    check.is_instance(proba_std, np.ndarray)
    check.is_true(proba_std.shape == (32, 32, 4))
    check.greater_equal(proba_std.min(), 0.0)
    # Case batch_size = 1
    proba_mean, _, _ = gen.proba_map(z, pixel_map, batch_size=1)
    check.is_true(proba_mean.shape == (32, 32, 4))
