"""Test /utils/train/random_utils.py."""

import random

import numpy as np
import torch

from utils.train.random_utils import set_global_seed


def test_set_global_seed() -> None:
    """Test set_global_seed."""
    set_global_seed(0)
    # Test the expected values for this particular seed.
    assert torch.randint(1000, size=(1,)) == 44
    assert np.random.randint(0, 1000) == 684
    assert random.randint(0, 1000) == 864
