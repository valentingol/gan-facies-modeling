"""Tests for utils/gan/uncond_sagan/modules.py."""
from typing import Tuple

import pytest_check as check
import torch

from utils.configs import GlobalConfig
from utils.gan.uncond_sagan.modules import (UncondSADiscriminator,
                                            UncondSAGenerator)


def test_sa_discriminator(configs: Tuple[GlobalConfig, GlobalConfig]) -> None:
    """Test SADiscriminator."""
    _, config64 = configs

    disc = UncondSADiscriminator(n_classes=4, model_config=config64.model)
    x = torch.rand(size=(1, 4, 64, 64), dtype=torch.float32)
    x = x / torch.sum(x, dim=1, keepdim=True)  # normalize
    preds, att_list = disc(x, with_attn=True)
    check.equal(preds.shape, (1,))
    check.equal(len(att_list), 1)
    check.equal(att_list[0].shape, (1, 16, 16))
    preds = disc(x, with_attn=False)
    check.is_instance(preds, torch.Tensor)


def test_sa_generator(configs: Tuple[GlobalConfig, GlobalConfig]) -> None:
    """Test SAGenerator."""
    _, config64 = configs

    gen = UncondSAGenerator(n_classes=4, model_config=config64.model)
    z = torch.rand(size=(1, 128), dtype=torch.float32)
    data, att_list = gen(z, with_attn=True)
    check.equal(data.shape, (1, 4, 64, 64))
    check.equal(len(att_list), 1)
    check.equal(att_list[0].shape, (1, 1024, 1024))
    data = gen(z, with_attn=False)
    check.is_instance(data, torch.Tensor)

    # Test generate method
    images, attn_list = gen.generate(z, with_attn=True)
    check.equal(images.shape, (1, 64, 64, 3))
    check.equal(len(attn_list), 1)
    images, attn_list = gen.generate(z, with_attn=False)
    check.equal(images.shape, (1, 64, 64, 3))
    check.equal(attn_list, [])
