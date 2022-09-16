"""Tests for utils/metrics/tools.py."""

import os
import shutil

import pytest
import pytest_check as check

from utils.metrics.tools import (MetricsType, get_n_classes, save_metrics,
                                 split_wass_dists)


@pytest.fixture
def metrics() -> MetricsType:
    """Return a metrics dict for tests."""
    w_dists = {}
    for cls_id in [1, 2]:
        for ind_name in ['ind1', 'ind2', 'ind3']:
            w_dists[f'{ind_name}_cls_{cls_id}'] = 0.1 * cls_id
    w_dists['global'] = 0.3
    other_metrics = {'cond_acc': 0.4}
    return w_dists, other_metrics


def test_get_n_classes(metrics: MetricsType) -> None:
    """Test get_n_classes."""
    check.equal(get_n_classes(metrics[0]), 2)


def test_split_wass_dists(metrics: MetricsType) -> None:
    """Test split_wass_dists."""
    split_wdists = split_wass_dists(metrics[0])
    check.equal(len(split_wdists), 3)
    inds_0 = split_wdists[0]
    inds_1 = split_wdists[1]
    for ind_name in ['ind1', 'ind2', 'ind3']:
        check.equal(inds_0[ind_name], 0.1)
        check.equal(inds_1[ind_name], 0.2)
    check.equal(split_wdists[2]['global'], 0.3)


def test_save_metrics(metrics: MetricsType) -> None:
    """Test save_metrics."""
    metrics_save_path = 'res/tmp_test/metrics/metrics'
    save_metrics(metrics, metrics_save_path, save_json=True, save_csv=True)
    check.is_true(os.path.exists(metrics_save_path + '.json'))
    check.is_true(os.path.exists(metrics_save_path + '.csv'))
    shutil.rmtree('res/tmp_test')
