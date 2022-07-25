"""Time utilities."""

import time
from typing import Tuple


def get_delta_eta(start_time: float, start_step: int, step: int,
                  total_step: int) -> Tuple[str, str]:
    """Return the delta and eta time under string format."""
    delta_t = int(time.time() - start_time)
    delta_str = (f'{delta_t // 3600:02d}h'
                 f'{(delta_t // 60) % 60:02d}m'
                 f'{delta_t % 60:02d}s')
    eta_t = ((total_step+start_step-step-1) * delta_t // (step+1-start_step))
    eta_str = (f'{eta_t // 3600:02d}h'
               f'{(eta_t // 60) % 60:02d}m'
               f'{eta_t % 60:02d}s')
    return delta_str, eta_str
