"""Tests for sagan/spectral.py."""

import torch

from utils.sagan.spectral import SpectralNorm, l2normalize


def test_l2normalize() -> None:
    """Test l2normalize."""
    vect = torch.rand(size=(10,), dtype=torch.float32)
    l2normalize(vect)


def test_spectral_norm() -> None:
    """Test SpectralNorm."""
    module = torch.nn.Conv2d(3, 8, kernel_size=3)
    weights = module.weight.data
    spec_norm = SpectralNorm(module, weight_name='weight')
    assert spec_norm._are_made_params()  # pylint: disable=protected-access
    assert hasattr(module, 'weight_u')
    assert hasattr(module, 'weight_v')
    assert hasattr(module, 'weight_bar')
    spec_norm(torch.rand(size=(1, 3, 32, 32)))
    new_weights = module.weight.data
    assert not torch.allclose(weights, new_weights)
