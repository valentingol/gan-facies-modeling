"""Utilities for randomness."""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set global seed for each source of randomness.

    Note
    ----
    This function **NOT ENSURE** perfect reproducibility
    across different devices but limit the differences in behavior
    between machines.
    """
    torch.manual_seed(seed)  # set for CPU and CUDA
    np.random.seed(seed)
    random.seed(seed)
