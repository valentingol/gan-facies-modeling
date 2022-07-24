"""Tests for utils/github_actions/pylint_manager.py."""

import os
import sys
from typing import Any

import pytest

from utils.github_actions.pylint_manager import check_output


def test_check_output() -> None:
    """Test check_output."""
    old_argv = sys.argv.copy()
    sys.argv = ['--score=9.0', '--score_min=8.0']
    assert check_output() == (9.0, 8.0)
    sys.argv = ['--score=7.0', '--score_min=8.0']
    with pytest.raises(ValueError,
                       match='.*score 7.0 is lower than minimum.*'):
        check_output()
    sys.argv = old_argv


def test_pylint_manager(capfd: Any) -> None:
    """Test pylint_manager script."""
    run = os.system('python utils/github_actions/pylint_manager.py'
                    ' --score=9.0 --score_min=8.0')
    assert run == 0
    out, _ = capfd.readouterr()
    assert out == '#ffff00\n'
    run = os.system('python utils/github_actions/pydocstyle_manager.py'
                    ' --score=7.0 --score_min=8.0')
    assert run != 0  # raise error
