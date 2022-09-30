"""Tests for utils/config.py."""

import sys

import pytest_check as check

from gan_facies.utils.configs import GlobalConfig, merge_configs


def test_global_config() -> None:
    """Test GlobalConfig."""
    old_argv = sys.argv.copy()  # Save sys.argv
    sys.argv = []  # Reset sys.argv
    config = GlobalConfig.build_from_argv(
        fallback='tests/configs/data64.yaml')
    check.equal(config.run_name, 'tmp_test')
    check.equal(config.model.data_size, 64)
    check.equal(config.training.total_time, -1)
    sys.argv = old_argv  # Restore sys.argv


def test_merge_configs() -> None:
    """Test merge_configs."""
    config = GlobalConfig.build_from_argv(
        fallback='tests/configs/data64.yaml')
    new_dict_config = {'model.data_size': 128}
    new_config = merge_configs(config, new_dict_config)
    check.equal(new_config.model.data_size, 128)
