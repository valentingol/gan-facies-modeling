"""Tests for utils/gan/cond_sagan/module.py."""
from typing import Tuple

import pytest_check as check
import torch

from utils.configs import GlobalConfig
from utils.gan.cond_sagan.modules import CondSAGenerator


def test_sa_generator(configs: Tuple[GlobalConfig, GlobalConfig]) -> None:
    """Test SAGenerator."""
    config32, _ = configs

    pixel_maps = torch.randint(0, 2, size=(5, 4, 32, 32), dtype=torch.float32)
    gen = CondSAGenerator(n_classes=4, model_config=config32.model)
    z = torch.rand(size=(5, 128), dtype=torch.float32)
    data, att_list = gen(z, pixel_maps, with_attn=True)
    check.equal(data.shape, (5, 4, 32, 32))
    check.equal(len(att_list), 3)
    check.equal(att_list[0].shape, (5, 16, 16))
    check.equal(att_list[1].shape, (5, 64, 64))
    check.equal(att_list[2].shape, (5, 256, 256))
    data = gen(z, pixel_maps, with_attn=False)
    check.is_instance(data, torch.Tensor)

    # Test generate method
    images, attn_list = gen.generate(z, pixel_maps, with_attn=True)
    check.equal(images.shape, (5, 32, 32, 3))
    check.equal(len(attn_list), 3)
    images, attn_list = gen.generate(z, pixel_maps, with_attn=False)
    check.equal(images.shape, (5, 32, 32, 3))
    check.equal(attn_list, [])
