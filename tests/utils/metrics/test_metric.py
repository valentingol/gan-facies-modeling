"""Test for utils/metrics/metric.py."""

import os
from typing import Any

import numpy as np
import pytest

from utils.metrics.metric import wasserstein_distances


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
        data1, data2, save_boxes_path='tests/tmp_boxes.png')
    assert os.path.exists('tests/tmp_boxes.png'), 'Save boxes failed.'
    os.remove('tests/tmp_boxes.png')
    some_expected_keys = {'prop_cls_1', 'proba_cls_1', 'adj_to_2_prop_cls_1',
                          'num_connected_cls_2', 'box_ratio_cls_2', 'global'}
    assert some_expected_keys <= w_dists.keys()
    for values in w_dists.values():
        assert isinstance(values, float)
    assert len(list1) == len(list2)

    # Case input indicators
    w_dists, (list1, list2) = wasserstein_distances(indicators_list_1,
                                                    indicators_list_2)
    assert (list1 == indicators_list_1) and (list2 == indicators_list_2)
    for values in w_dists.values():
        assert isinstance(values, float)
    assert w_dists.keys() == {'ind1_cls_1', 'ind2_cls_1', 'ind1_cls_2',
                              'ind2_cls_2', 'global'}

    # Case wrong inputs
    with pytest.raises(TypeError, match='.*data1.*'):
        wasserstein_distances('indicators_list_1',  # type: ignore
                              indicators_list_2)
    with pytest.raises(TypeError, match='.*data2.*'):
        wasserstein_distances(indicators_list_1,
                              'indicators_list_2')  # type: ignore

    # Case indicators with different length
    wasserstein_distances(indicators_list_1, indicators_list_2[:-1])
    captured = capsys.readouterr()
    assert captured.out.startswith('Warning: data1 and data2 have different '
                                   'number of classes')
