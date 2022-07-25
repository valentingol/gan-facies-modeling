"""Tests for utils/github_actions/pydocstyle_manager.py."""

import os
import sys

import pytest

from utils.github_actions.pydocstyle_manager import check_output


def test_check_output() -> None:
    """Test check_output."""
    old_argv = sys.argv.copy()
    sys.argv = ['--n_errors=0']
    check_output()
    sys.argv = ['--n_errors=1']
    with pytest.raises(ValueError, match='.*found 1 error.*'):
        check_output()
    sys.argv = old_argv


def test_pydocstyle_manager() -> None:
    """Test pydocstyle_manager script."""
    run = os.system('python utils/github_actions/pydocstyle_manager.py'
                    ' --n_errors=0')
    assert run == 0  # raise no error
    run = os.system('python utils/github_actions/pydocstyle_manager.py'
                    ' --n_errors=1')
    assert run != 0  # raise error
