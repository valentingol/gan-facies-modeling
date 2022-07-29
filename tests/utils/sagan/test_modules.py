"""Tests for sagan/modules.py."""

from typing import Tuple

import pytest
import torch

from utils.configs import GlobalConfig
from utils.sagan.modules import SADiscriminator, SAGenerator, SelfAttention


@pytest.fixture
def configs() -> Tuple[GlobalConfig, GlobalConfig]:
    """Return configs with data size 32 and 64"""
    config32 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data32.yaml')
    config64 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data64.yaml')
    return config32, config64


def test_self_attention() -> None:
    """Test SelfAttention."""
    self_att = SelfAttention(in_dim=200)
    x = torch.rand(size=(5, 200, 16, 16), dtype=torch.float32)
    out, att = self_att(x)
    assert out.shape == (5, 200, 16, 16)
    assert att.shape == (5, 256, 256)
    assert torch.min(att) >= 0
    assert torch.allclose(torch.sum(att, dim=-1),
                          torch.tensor(1, dtype=torch.float32))


def test_sa_discriminator(configs: Tuple[GlobalConfig, GlobalConfig]) -> None:
    """Test SADiscriminator."""
    config32, config64 = configs

    # config 32
    disc = SADiscriminator(n_classes=4, model_config=config32.model)
    x = torch.rand(size=(5, 4, 32, 32), dtype=torch.float32)
    x = x / torch.sum(x, dim=1, keepdim=True)  # normalize
    preds, att_list = disc(x)
    assert len(att_list) == 1
    assert att_list[0].shape == (5, 16, 16)
    assert preds.shape == (5,)

    # config 64
    disc = SADiscriminator(n_classes=4, model_config=config64.model)
    x = torch.rand(size=(1, 4, 64, 64), dtype=torch.float32)
    x = x / torch.sum(x, dim=1, keepdim=True)  # normalize
    preds, att_list = disc(x)
    assert preds.shape == (1,)
    assert len(att_list) == 2
    assert att_list[0].shape == (1, 64, 64)
    assert att_list[1].shape == (1, 16, 16)


def test_sa_generator(configs: Tuple[GlobalConfig, GlobalConfig]) -> None:
    """Test SAGenerator."""
    config32, config64 = configs

    # config 32
    gen = SAGenerator(n_classes=4, model_config=config32.model)
    z = torch.rand(size=(5, 128), dtype=torch.float32)
    data, att_list = gen(z)
    assert data.shape == (5, 4, 32, 32)
    assert len(att_list) == 1
    assert att_list[0].shape == (5, 256, 256)

    # config 64
    gen = SAGenerator(n_classes=4, model_config=config64.model)
    z = torch.rand(size=(1, 128), dtype=torch.float32)
    data, att_list = gen(z)
    assert data.shape == (1, 4, 64, 64)
    assert len(att_list) == 2
    assert att_list[0].shape == (1, 256, 256)
    assert att_list[1].shape == (1, 1024, 1024)

    # Test generate method
    images, _ = gen.generate(z, with_attn=True)
    assert images.shape == (1, 64, 64, 3)
    images, attn_list = gen.generate(z, with_attn=False)
    assert images.shape == (1, 64, 64, 3)
    assert not attn_list
