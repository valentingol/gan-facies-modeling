"""Tests for utils/metrics/tools.py."""

import os
import shutil
from typing import Dict

import pytest
import pytest_check as check

from utils.metrics.tools import get_n_classes, save_metrics, split_metric


@pytest.fixture
def metrics() -> Dict[str, float]:
    """Return a metrics dict for tests."""
    metrics = {}
    for cls_id in [1, 2]:
        for ind_name in ['ind1', 'ind2', 'ind3']:
            metrics[f'{ind_name}_cls_{cls_id}'] = 0.1 * cls_id
    metrics['global'] = 0.3
    return metrics


def test_get_n_classes(metrics: Dict[str, float]) -> None:
    """Test get_n_classes."""
    check.equal(get_n_classes(metrics), 2)


def test_split_metric(metrics: Dict[str, float]) -> None:
    """Test split_metric."""
    split_metrics = split_metric(metrics)
    check.equal(len(split_metrics), 3)
    inds_0 = split_metrics[0]
    inds_1 = split_metrics[1]
    for ind_name in ['ind1', 'ind2', 'ind3']:
        check.equal(inds_0[ind_name], 0.1)
        check.equal(inds_1[ind_name], 0.2)
    check.equal(split_metrics[2]['global'], 0.3)


def test_save_metrics(metrics: Dict[str, float]) -> None:
    """Test save_metrics."""
    metrics_save_path = 'res/tmp_test/metrics/metrics'
    save_metrics(metrics, metrics_save_path, save_json=True, save_csv=True)
    check.is_true(os.path.exists(metrics_save_path + '.json'))
    check.is_true(os.path.exists(metrics_save_path + '.csv'))
    shutil.rmtree('res/tmp_test')
