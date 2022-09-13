"""Test for utils/metrics/visualize.py."""

from typing import Dict, List

import numpy as np
import pytest
import pytest_check as check
from matplotlib import pyplot as plt

from utils.metrics.visualize import plot_boxes


@pytest.fixture
def indicators_list() -> List[Dict[str, List[float]]]:
    """Return a small list of indicators for testing."""
    indicators_1 = {
        'ind1': np.random.rand(6).tolist(),
        'ind2': np.random.rand(6).tolist(),
    }
    indicators_2 = {
        'ind1': np.random.rand(6).tolist(),
        'ind3': np.random.rand(6).tolist(),  # ind2 replaced by ind3
    }
    return [indicators_1, indicators_2]


def test_plot_boxs(indicators_list: List[Dict[str, List[float]]]) -> None:
    """Test plot_boxes function."""
    fig = plt.figure()
    plot_boxes(indicators_list, indicator_names=None, indicator_colors=None)
    check.is_true(fig.get_axes(), 'No plot found.')
    fig = plt.figure()
    plot_boxes(indicators_list, indicator_names=['name1', 'name2'],
               indicator_colors=['lightblue', 'lightgreen'])
    check.is_true(fig.get_axes(), 'No plot found.')
