"""Common fixtures and check functions for tests/utils."""
import os
from typing import Tuple

import numpy as np
import pytest
from pytest_check import check_func

from utils.configs import GlobalConfig


@check_func
def check_allclose(arr1: np.ndarray, arr2: np.ndarray) -> None:
    """Check if two arrays are all close."""
    assert np.allclose(arr1, arr2)


@check_func
def check_exists(path: str) -> None:
    """Check if a path exists."""
    assert os.path.exists(path)


@pytest.fixture
def configs() -> Tuple[GlobalConfig, GlobalConfig]:
    """Return configs with data size 32 and 64."""
    config32 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data32.yaml')
    config64 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data64.yaml')
    return config32, config64
