"""Tests for utils/gan/attention.py."""

import pytest_check as check
import torch

from gan_facies.gan.attention import SelfAttention
from gan_facies.utils.configs import ConfigType


def test_self_attention(configs: ConfigType) -> None:
    """Test SelfAttention."""
    config_32, config_64 = configs

    # Case with out_layer = True
    attention = SelfAttention(in_dim=16,
                              attention_config=config_32.model.attention)
    out, attention = attention(torch.rand(2, 16, 9, 9))
    check.equal(out.shape, (2, 16, 9, 9))
    check.equal(attention.shape, (2, 1, 81, 81))

    # Case with out_layer = False
    attention = SelfAttention(in_dim=16,
                              attention_config=config_64.model.attention)
    out, attention = attention(torch.rand(2, 16, 9, 9))
    check.equal(out.shape, (2, 16, 9, 9))
    check.equal(attention.shape, (2, 4, 81, 81))

    # Case with out_layer = False and v_ratio != 1
    config_64_bis = config_64.copy()
    config_64_bis.merge({"model": {"attention": {"v_ratio": 2}}},
                        do_not_pre_process=True)
    with check.raises(ValueError):
        SelfAttention(in_dim=16,
                      attention_config=config_64_bis.model.attention)
