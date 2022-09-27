"""Tests for utils.gan.initialization.py."""

import pytest_check as check
import torch
from torch import nn

from utils.gan.initialization import init_weights


def test_init_weights() -> None:
    """Test init_weights."""
    module = nn.Sequential(nn.Conv2d(3, 3, 3), nn.Conv2d(3, 3, 3))
    weights_1 = module[0].weight.clone()
    init_weights(module, 'default')
    weights_2 = module[0].weight.clone()
    check.is_true(torch.allclose(weights_1, weights_2))
    init_weights(module, 'orthogonal')
    weights_3 = module[0].weight.clone()
    check.is_false(torch.allclose(weights_2, weights_3))
    init_weights(module, 'glorot')
    weights_4 = module[0].weight.clone()
    check.is_false(torch.allclose(weights_3, weights_4))
    init_weights(module, 'normal')
    weights_5 = module[0].weight.clone()
    check.is_false(torch.allclose(weights_4, weights_5))
    with check.raises(ValueError):
        init_weights(module, 'UNKNOWN')
