"""Tests for utils/post_process.py."""

import numpy as np
import pytest_check as check
from pytest_mock import MockerFixture

from gan_facies.utils.post_process import (clip_by_indicator,
                                           clip_indicator_eval, erode_dilate,
                                           fill_blank_neighborhood,
                                           postprocess_erode_dilate)

# Clip by indicator functions


def test_clip_by_indicator() -> None:
    """Test clip_by_indicator."""
    data = np.random.randn(10, 32, 32, 3)
    range10 = [float(k) for k in range(10)]
    indicators = [{'ind1': range10, 'ind2': range10[::-1]},
                  {'ind1': range10[::-1], 'ind2': range10}]
    data_new, indicators_new = clip_by_indicator(data, indicators, 'ind1',
                                                 0, 0.2, 'above')
    check.is_true(np.array_equal(data_new, data[2:]))
    check.equal(indicators_new[1]['ind2'], list(range(2, 10)))

    data_new, indicators_new = clip_by_indicator(data, indicators, 'ind2',
                                                 1, 0.2, 'below')
    check.is_true(np.array_equal(data_new, data[:2]))
    check.equal(indicators_new[0]['ind2'], [9, 8])


def test_clip_indicator_eval(mocker: MockerFixture) -> None:
    """Test clip_indicator_eval."""
    mocker.patch('gan_facies.metrics.metric.wasserstein_distances',
                 side_effect=lambda *args, **kwargs: [{'global': 0.3}])
    data = np.random.randn(10, 32, 32, 3)
    range10 = [float(k) for k in range(10)]
    indicators_ref = [{'ind1': range10, 'ind2': range10[::-1]},
                      {'ind1': range10[::-1], 'ind2': range10}]
    clip_by_indicator_params = {'data': data,
                                'indicators': indicators_ref.copy(),
                                'indicator_name': 'ind1',
                                'class_id': 0,
                                'alpha': 0.2,
                                'order': 'above'}
    metric_params = {'connectivity': 2, 'unit_component_size': 2}

    data_new, indicators_new, length_new, dists_new = clip_indicator_eval(
        indicators_ref, clip_by_indicator_params, metric_params
        )
    check.is_true(np.array_equal(data_new, data[2:]))
    check.equal(indicators_new[1]['ind2'], list(range(2, 10)))
    check.equal(length_new, 8)
    check.equal(dists_new, {'global': 0.3})


# Erosion/dilation functions

def test_erode_dilate() -> None:
    """Test erode/dilate."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:15, 10:15] = 1
    mask[15, 15] = mask[15, 16] = mask[16, 17] = 1
    mask[0, 2] = mask[2, 2] = 1
    new_mask = erode_dilate(mask)
    expected_mask = np.zeros((32, 32), dtype=np.uint8)
    expected_mask[10:15, 10:15] = 1
    expected_mask[15, 15] = expected_mask[15, 16] = expected_mask[16, 17] = 1
    check.is_true(np.array_equal(new_mask, expected_mask))


def test_fill_blank_neighborhood() -> None:
    """Test fill_blank_neighborhood."""
    img = np.zeros((32, 32), dtype=np.int32)
    # 5/8 1-neighbors
    img[8, 12] = img[8, 13] = img[8, 14] = img[10, 12] = img[10, 14] = 1
    img[9, 13] = -1
    new_img = fill_blank_neighborhood(img)
    check.equal(new_img[9, 13], 1)
    # 3/8 1-neighbors
    img[8, 12] = img[8, 13] = 0
    img[9, 13] = -1
    new_img = fill_blank_neighborhood(img)
    check.equal(new_img[9, 13], 0)


def test_postprocess_erode_dilate() -> None:
    """Test postprocess_erode_dilate."""
    img = np.zeros((32, 32), dtype=np.uint8)
    img[10:15, 10:15] = 1
    img[20:25, 20:25] = 2
    new_img = postprocess_erode_dilate(img)
    check.equal(new_img.max(), 2)
    check.equal(new_img.min(), 0)
