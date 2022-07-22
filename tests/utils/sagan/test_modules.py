"""Tests for sagan/modules.py."""

import torch

from utils.sagan.modules import SADiscriminator, SAGenerator, SelfAttention


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


def test_sa_discriminator() -> None:
    """Test SADiscriminator."""
    # data_size = 32, batch_size = 5
    disc = SADiscriminator(n_classes=4, data_size=32, conv_dim=64)
    x = torch.rand(size=(5, 4, 32, 32), dtype=torch.float32)
    x = x / torch.sum(x, dim=1, keepdim=True)  # normalize
    preds, att_list = disc(x)
    assert len(att_list) == 1
    assert att_list[0].shape == (5, 16, 16)
    assert preds.shape == (5,)

    # data_size = 64, batch_size = 1
    disc = SADiscriminator(n_classes=4, data_size=64, conv_dim=64)
    x = torch.rand(size=(1, 4, 64, 64), dtype=torch.float32)
    x = x / torch.sum(x, dim=1, keepdim=True)  # normalize
    preds, att_list = disc(x)
    assert preds.shape == (1,)
    assert len(att_list) == 2
    assert att_list[0].shape == (1, 64, 64)
    assert att_list[1].shape == (1, 16, 16)


def test_sa_generator() -> None:
    """Test SAGenerator."""
    # data_size = 32, batch_size = 5
    gen = SAGenerator(n_classes=4, data_size=32, z_dim=128, conv_dim=64)
    z = torch.rand(size=(5, 128), dtype=torch.float32)
    data, att_list = gen(z)
    assert data.shape == (5, 4, 32, 32)
    assert len(att_list) == 1
    assert att_list[0].shape == (5, 256, 256)
    # data_size = 64, batch_size = 1
    gen = SAGenerator(n_classes=4, data_size=64, z_dim=128, conv_dim=64)
    z = torch.rand(size=(1, 128), dtype=torch.float32)
    data, att_list = gen(z)
    assert data.shape == (1, 4, 64, 64)
    assert len(att_list) == 2
    assert att_list[0].shape == (1, 256, 256)
    assert att_list[1].shape == (1, 1024, 1024)
