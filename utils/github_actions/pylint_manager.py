"""Manage Pylint output on workflow."""
import sys

from utils.github_actions.color import score_to_hex_color


def check_output() -> float:
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
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith('--score='):
            score = float(arg.split('=')[1])

    if score < SCORE_MIN:
        raise ValueError(
                f'Pylint score {score} is lower than '
                f'minimum ({SCORE_MIN})'
                )

    return score


if __name__ == '__main__':
    # SCORE_MIN can be changed safely depending on your needs.
    SCORE_MIN = 7.0
    SCORE_MAX = 10.0

    score = check_output()

    # Print color to be used in GitHub Actions
    print(score_to_hex_color(score, SCORE_MIN, SCORE_MAX))
