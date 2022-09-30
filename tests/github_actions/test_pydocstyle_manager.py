"""Tests for utils/github_actions/pydocstyle_manager.py."""

import os
import sys

import pytest_check as check

from github_actions.pydocstyle_manager import check_output


def test_check_output() -> None:
    """Test check_output."""
    old_argv = sys.argv.copy()  # Save sys.argv
    sys.argv = ['--n_errors=0']
    check_output()
    sys.argv = ['--n_errors=1']
    with check.raises(ValueError):
        check_output()
    sys.argv = old_argv  # Restore sys.argv


def test_pydocstyle_manager() -> None:
    """Test pydocstyle_manager script."""
    run = os.system('python github_actions/pydocstyle_manager.py'
                    ' --n_errors=0')
    check.equal(run, 0)  # raise no error
    run = os.system('python github_actions/pydocstyle_manager.py'
                    ' --n_errors=1')
    check.not_equal(run, 0)  # raise error
