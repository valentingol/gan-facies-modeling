"""Test for utils/metrics/metric.py."""

import os
import shutil
from typing import Any, Tuple

import numpy as np
import pytest_check as check

from tests.utils.conftest import check_exists
from tests.utils.gan.test_base_trainer import DataLoader32, DataLoader64
from utils.configs import GlobalConfig
from utils.gan.cond_sagan.modules import CondSAGenerator
from utils.gan.uncond_sagan.modules import UncondSAGenerator
from utils.metrics.metric import (compute_save_indicators, evaluate,
                                  print_metrics, wasserstein_distances)


def test_wasserstein_distances(capsys: Any) -> None:
    """Test wasserstein_distances."""
    data1 = np.random.randint(0, 3, size=(5, 32, 32), dtype=np.uint8)
    data1[0, 0, :2] = [1, 2]  # Force to have 2 classes (in addition to 0)
    data2 = np.random.randint(0, 3, size=(5, 32, 32), dtype=np.uint8)
    data2[0, 0, :2] = [1, 2]  # Force to have 2 classes (in addition to 0)
    indicators_list_1 = [{
        'ind1': np.random.rand(5).tolist(),
        'ind2': np.random.rand(5).tolist(),
    } for _ in range(2)]
    indicators_list_2 = [{
        'ind1': np.random.rand(5).tolist(),
        'ind2': np.random.rand(5).tolist(),
    } for _ in range(2)]

    # Case input data + save boxes
    w_dists, (list1, list2) = wasserstein_distances(
        data1, data2, connectivity=None, unit_component_size=1,
        save_boxes_path='tests/tmp_boxes.png')
    check_exists('tests/tmp_boxes.png')
    os.remove('tests/tmp_boxes.png')
    some_expected_keys = {
        'prop_cls_1', 'proba_cls_1', 'adj_to_2_prop_cls_1',
        'num_connected_cls_2', 'box_ratio_cls_2', 'global'
    }
    check.less_equal(some_expected_keys, w_dists.keys())
    for values in w_dists.values():
        check.is_instance(values, float)
    check.equal(len(list1), len(list2))

    # Case input indicators
    w_dists, (list1, list2) = wasserstein_distances(indicators_list_1,
                                                    indicators_list_2,
                                                    connectivity=None,
                                                    unit_component_size=1)
    check.equal(list1, indicators_list_1)
    check.equal(list2, indicators_list_2)
    for values in w_dists.values():
        check.is_instance(values, float)
    check.equal(w_dists.keys(),
                {'ind1_cls_1', 'ind2_cls_1', 'ind1_cls_2',
                 'ind2_cls_2', 'global'})

    # Case wrong inputs
    with check.raises(TypeError):
        wasserstein_distances('indicators_list_1',  # type: ignore
                              indicators_list_2,
                              connectivity=None,
                              unit_component_size=1)
    with check.raises(TypeError):
        wasserstein_distances(indicators_list_1,
                              'indicators_list_2',  # type: ignore
                              connectivity=None,
                              unit_component_size=1)

    # Case indicators with different length
    wasserstein_distances(indicators_list_1, indicators_list_2[:-1],
                          connectivity=None, unit_component_size=1)
    wasserstein_distances(indicators_list_1[:-1], indicators_list_2,
                          connectivity=None, unit_component_size=1)
    captured = capsys.readouterr()
    check.is_true(captured.out.startswith('Warning: data1 and data2 have '
                                          'different number of classes'))


def test_compute_save_indicators(capsys: Any) -> None:
    """Test compute_save_indicators."""
    data_loader = DataLoader32()
    config32 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data32.yaml')
    data32 = np.random.randint(0, 3, size=(4, 32, 32), dtype=np.uint8)
    os.makedirs('tests/datasets', exist_ok=True)
    np.save('tests/datasets/data32.npy', data32)

    # Case with overwrite_indicators = True
    indicators_path = compute_save_indicators(data_loader, config32)
    expected_path = 'tests/datasets/data32_ds32_co2_us4_indicators.json'
    check.equal(indicators_path, expected_path)
    check_exists(indicators_path)

    # Case with overwrite_indicators = False
    config32.merge({'overwrite_indicators': False})
    check.is_(config32.overwrite_indicators, False)
    compute_save_indicators(data_loader, config32)
    captured = capsys.readouterr()
    check.is_in('re-used', captured.out)

    os.remove(indicators_path)
    os.remove('tests/datasets/data32.npy')


def test_evaluate(configs: Tuple[GlobalConfig, GlobalConfig]) -> None:
    """Test evaluate."""
    config32, config64 = configs
    data_loader32 = DataLoader32()
    data_loader64 = DataLoader64()
    data32 = np.random.randint(0, 3, size=(4, 32, 32), dtype=np.uint8)
    data64 = np.random.randint(0, 3, size=(4, 64, 64), dtype=np.uint8)
    os.makedirs('tests/datasets', exist_ok=True)
    np.save('tests/datasets/data32.npy', data32)
    np.save('tests/datasets/data64.npy', data64)
    gen32 = CondSAGenerator(n_classes=3, model_config=config32.model)
    # Case indicators are missing
    with check.raises(FileNotFoundError):
        evaluate(gen32, config32, training=True, step=8, indicators_path=None,
                 save_json=True, save_csv=True, n_images=10)

    # Case unconditional and training = False

    # Save indicators
    indicators_path = compute_save_indicators(data_loader32, config32)

    evaluate(gen32, config32, training=False, step=13, indicators_path=None,
             save_json=True, save_csv=True, n_images=4)
    check_exists('res/tmp_test/metrics/test_boxes_step_13.png')
    check_exists('res/tmp_test/metrics/test_metrics_step_13.json')
    check_exists('res/tmp_test/metrics/test_metrics_step_13.csv')
    os.remove(indicators_path)

    # Case conditional and training = True
    gen64 = UncondSAGenerator(n_classes=3, model_config=config64.model)

    # Save indicators
    indicators_path = compute_save_indicators(data_loader64, config64)

    evaluate(gen64, config64, training=True, step=8, indicators_path=None,
             save_json=True, save_csv=True, n_images=4)
    check_exists('res/tmp_test/metrics/boxes_step_8.png')
    check_exists('res/tmp_test/metrics/metrics_step_8.json')
    check_exists('res/tmp_test/metrics/metrics_step_8.csv')

    os.remove('tests/datasets/data32.npy')
    os.remove('tests/datasets/data64.npy')

    shutil.rmtree('res/tmp_test')


def test_print_metrics() -> None:
    """Test print_metrics."""
    metrics = {
        'ind1_cls_1': 0.3,
        'ind2_cls_1': 0.2,
        'ind1_cls_2': 0.1,
        'ind2_cls_2': 0.4,
        'global': 0.5,
    }
    print_metrics(metrics, step=None)
    print_metrics(metrics, step=2000)
