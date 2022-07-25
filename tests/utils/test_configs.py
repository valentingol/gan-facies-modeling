"""Tests for utils/config.py."""

import sys

from utils.configs import GlobalConfig


def test_global_config() -> None:
    """Test GlobalConfig."""
    sys.argv = []  # Reset sys.argv
    config = GlobalConfig.build_from_argv(
        fallback='configs/unittest/data64.yaml')
    assert config.run_name == 'tmp_test'
    assert config.model.data_size == 64
    assert config.training.total_time == -1
