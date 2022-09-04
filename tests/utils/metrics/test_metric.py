"""Test for utils/metrics/metric.py."""

import os
import shutil
from typing import Any

import numpy as np
import pytest

from tests.utils.sagan.test_trainer import DataLoader32
from utils.configs import GlobalConfig
from utils.metrics.metric import (compute_save_indicators, evaluate,
                                  print_metrics, wasserstein_distances)
from utils.sagan.modules import SAGenerator


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
    assert os.path.exists('tests/tmp_boxes.png'), 'Save boxes failed.'
    os.remove('tests/tmp_boxes.png')
    some_expected_keys = {
        'prop_cls_1', 'proba_cls_1', 'adj_to_2_prop_cls_1',
        'num_connected_cls_2', 'box_ratio_cls_2', 'global'
    }
    assert some_expected_keys <= w_dists.keys()
    for values in w_dists.values():
        assert isinstance(values, float)
    assert len(list1) == len(list2)

    # Case input indicators
    w_dists, (list1, list2) = wasserstein_distances(indicators_list_1,
                                                    indicators_list_2,
                                                    connectivity=None,
                                                    unit_component_size=1)
    assert (list1 == indicators_list_1) and (list2 == indicators_list_2)
    for values in w_dists.values():
        assert isinstance(values, float)
    assert w_dists.keys() == {
        'ind1_cls_1', 'ind2_cls_1', 'ind1_cls_2', 'ind2_cls_2', 'global'
    }

    # Case wrong inputs
    with pytest.raises(TypeError, match='.*data1.*'):
        wasserstein_distances('indicators_list_1',  # type: ignore
                              indicators_list_2,
                              connectivity=None,
                              unit_component_size=1)
    with pytest.raises(TypeError, match='.*data2.*'):
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
    assert captured.out.startswith('Warning: data1 and data2 have different '
                                   'number of classes')


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
    assert indicators_path == expected_path
    assert os.path.exists(indicators_path)

    # Case with overwrite_indicators = False
    config32.merge({'overwrite_indicators': False})
    assert config32.overwrite_indicators is False
    compute_save_indicators(data_loader, config32)
    captured = capsys.readouterr()
    assert 're-used' in captured.out

    os.remove(indicators_path)
    os.remove('tests/datasets/data32.npy')


def test_evaluate() -> None:
    """Test evaluate."""
    data_loader = DataLoader32()
    config32 = GlobalConfig().build_from_argv(
        fallback='configs/unittest/data32.yaml')
    data32 = np.random.randint(0, 3, size=(4, 32, 32), dtype=np.uint8)
    os.makedirs('tests/datasets', exist_ok=True)
    np.save('tests/datasets/data32.npy', data32)
    gen = SAGenerator(n_classes=3, model_config=config32.model)
    # Case indicators are missing
    with pytest.raises(FileNotFoundError, match='.*not found.*'):
        evaluate(gen, config32, training=True, step=8, indicators_path=None,
                 save_json=True, save_csv=True, n_images=4)

    # Save indicators
    indicators_path = compute_save_indicators(data_loader, config32)

    # Case training = True
    evaluate(gen, config32, training=True, step=8, indicators_path=None,
             save_json=True, save_csv=True, n_images=4)
    assert os.path.exists('res/tmp_test/metrics/boxes_step_8.png')
    assert os.path.exists('res/tmp_test/metrics/metrics_step_8.json')
    assert os.path.exists('res/tmp_test/metrics/metrics_step_8.csv')

    # Case training = False
    evaluate(gen, config32, training=False, step=13, indicators_path=None,
             save_json=True, save_csv=True, n_images=4)
    assert os.path.exists('res/tmp_test/metrics/test_boxes_step_13.png')
    assert os.path.exists('res/tmp_test/metrics/test_metrics_step_13.json')
    assert os.path.exists('res/tmp_test/metrics/test_metrics_step_13.csv')

    os.remove('tests/datasets/data32.npy')
    os.remove(indicators_path)
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
