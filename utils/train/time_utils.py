"""Time utilities."""

import time
from typing import Tuple


def get_delta_eta(start_time: float, start_step: int, step: int,
                  total_step: int) -> Tuple[str, str]:
    """Return the delta and eta time under string format."""

    def time_to_str(time: float) -> str:
        """Convert time in seconds to string format."""
        return (f'{time // 3600:02d}h'
                f'{(time // 60) % 60:02d}m'
                f'{time % 60:02d}s')

    delta_t = int(time.time() - start_time)
    delta_str = time_to_str(delta_t)
    eta_t = (total_step-step-1) * delta_t // (step+1-start_step)
    eta_str = time_to_str(eta_t)
    return delta_str, eta_str
