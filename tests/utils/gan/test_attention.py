"""Tests for utils/gan/attention.py."""

import pytest_check as check
import torch

from utils.gan.attention import SelfAttention


def test_self_attention() -> None:
    """Test SelfAttention."""
    attention = SelfAttention(in_dim=16, att_dim=8, full_values=False)
    out, attention = attention(torch.rand(2, 16, 9, 9))
    check.equal(out.shape, (2, 16, 9, 9))
    check.equal(attention.shape, (2, 81, 81))
    attention = SelfAttention(in_dim=16, att_dim=None, full_values=True)
    out, attention = attention(torch.rand(2, 16, 9, 9))
    check.equal(out.shape, (2, 16, 9, 9))
    check.equal(attention.shape, (2, 81, 81))
