"""Tests for utils/github_actions/pylint_manager.py."""

import os
import sys
from typing import Any

import pytest_check as check

from github_actions.pylint_manager import check_output


def test_check_output() -> None:
    """Test check_output."""
    old_argv = sys.argv.copy()  # Save sys.argv
    sys.argv = ['--score=9.0', '--score_min=8.0']
    check.equal(check_output(), (9.0, 8.0))
    sys.argv = ['--score=7.0', '--score_min=8.0']
    with check.raises(ValueError):
        check_output()
    sys.argv = old_argv  # Restore sys.argv


def test_pylint_manager(capfd: Any) -> None:
    """Test pylint_manager script."""
    run = os.system('python github_actions/pylint_manager.py'
                    ' --score=9.0 --score_min=8.0')
    check.equal(run, 0)
    out, _ = capfd.readouterr()
    check.equal(out, '#ffff00\n')
    run = os.system('python github_actions/pydocstyle_manager.py'
                    ' --score=7.0 --score_min=8.0')
    check.not_equal(run, 0)  # raise error
