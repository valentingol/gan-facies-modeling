"""Test /utils/train/time_utils.py."""

import time

from utils.train.time_utils import get_delta_eta


def test_get_delta_eta() -> None:
    """Test get_delta_eta."""
    params = {'start_step': 1000, 'step': 2000, 'total_step': 3000}
    start_time = time.time() - 90  # simulate 1 minute 30 seconds elapsed
    delta_str, eta_str = get_delta_eta(start_time=start_time, **params)
    # NOTE: add margin of error to taking into account the call time
    assert delta_str in [f'00h01m{i}s' for i in range(25, 35)]
    assert eta_str in [f'00h01m{i}s' for i in range(25, 35)]
