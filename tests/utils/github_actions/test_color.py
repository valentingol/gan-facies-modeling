"""Tests for utils/github_actions/color.py."""

import pytest_check as check

from utils.github_actions.color import score_to_hex_color


def test_score_to_hex_color() -> None:
    """Test score_to_hex_color."""
    res6 = score_to_hex_color(score=6.0, score_min=6.0, score_max=10.0)
    res8 = score_to_hex_color(score=8.0, score_min=6.0, score_max=10.0)
    res10 = score_to_hex_color(score=10.0, score_min=6.0, score_max=10.0)
    check.equal(res6, '#ff0000')
    check.equal(res8, '#ffff00')
    check.equal(res10, '#00ff00')
