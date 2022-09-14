"""Tests for /utils/auxiliaries.py."""

import random
import time

import numpy as np
import pytest_check as check
import torch

from utils.auxiliaries import get_delta_eta, set_global_seed


def test_set_global_seed() -> None:
    """Test set_global_seed."""
    set_global_seed(0)
    # Test the expected values for this particular seed.
    check.equal(torch.randint(1000, size=(1,)), 44)
    check.equal(np.random.randint(0, 1000), 684)
    check.equal(random.randint(0, 1000), 864)


def test_get_delta_eta() -> None:
    """Test get_delta_eta."""
    params = {'start_step': 1000, 'step': 2000, 'total_step': 3000}
    start_time = time.time() - 90  # simulate 1 minute 30 seconds elapsed
    delta_str, eta_str = get_delta_eta(start_time=start_time, **params)
    # NOTE: add margin of error to taking into account the call time
    check.is_in(delta_str, [f'00h01m{i}s' for i in range(25, 35)])
    check.is_in(eta_str, [f'00h01m{i}s' for i in range(25, 35)])
