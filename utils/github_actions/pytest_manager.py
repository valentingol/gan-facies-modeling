"""Manage Pytest-cov output on workflow."""

import sys

from utils.github_actions.color import score_to_hex_color


def check_output() -> float:
    """Check output of Pytest-cov.

    Raises
    ------
    ValueError
        If Pytest find failures.
    ValueError
        If coverage is below SCORE_MIN.

    Returns
    -------
    score: float
        Score of coverage.
    """
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith('--score='):
            score_percent = arg.split('=')[1]
            score = float(score_percent.split('%')[0])
        if arg.startswith('--n_failures='):
            n_failures_str = arg.split('=')[1]
            if n_failures_str == '':
                n_failures = 0
            else:
                n_failures = int(n_failures_str)

    if n_failures > 0:
        raise ValueError(f'Pytest finds {n_failures} failure(s) on tests.')

    if score < SCORE_MIN:
        raise ValueError(
                f'Pytest coverage {score}% is lower than '
                f'minimum ({SCORE_MIN}%)'
                )

    return score


if __name__ == '__main__':
    # SCORE_MIN can be changed safely depending on your needs.
    # NOTE: score on %
    SCORE_MIN = 0
    SCORE_MAX = 100

    score = check_output()

    # Print color to be used in GitHub Actions
    print(score_to_hex_color(score, SCORE_MIN, SCORE_MAX))
