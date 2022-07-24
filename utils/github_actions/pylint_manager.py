"""Manage Pylint output on workflow."""
import sys
from typing import Tuple

from utils.github_actions.color import score_to_hex_color


def check_output() -> Tuple[float, float]:
    """Check output of Pylint.

    Raises
    ------
    ValueError
        If Pylint score is below SCORE_MIN.

    Returns
    -------
    score: float
        Score of Pylint.
    """
    args = sys.argv
    for arg in args:
        if arg.startswith('--score='):
            score = float(arg.split('=')[1])
        elif arg.startswith('--score_min='):
            score_min = float(arg.split('=')[1])

    if score < score_min:
        raise ValueError(
                f'Pylint score {score} is lower than '
                f'minimum ({score_min}).'
                )

    return score, score_min


if __name__ == '__main__':
    SCORE_MAX = 10.0

    score, score_min = check_output()  # raise error if score < score_min

    # Print color to be used in GitHub Actions
    print(score_to_hex_color(score, score_min, SCORE_MAX))
