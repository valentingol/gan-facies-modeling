"""Test for utils/metrics/indicators.py."""

from typing import Tuple

import numpy as np
import pytest

from utils.metrics.indicators import compute_indicators


@pytest.fixture
def data() -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D and 3D small random data for testing."""
    data_2d = np.random.randint(0, 4, size=(5, 10, 10), dtype=np.uint8)
    data_2d[0] = 0  # no object to test this limit case
    data_3d = np.random.randint(0, 4, size=(5, 5, 5, 5), dtype=np.uint8)
    return data_2d, data_3d


def test_compute_indicators(data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test compute_indicators."""
    data_2d, data_3d = data
    # Try different connectivity, unit_component_size and normalization
    indicators_list_1 = compute_indicators(data_2d, connectivity=2,
                                           unit_component_size=1,
                                           normalization='none')
    indicators_list_2 = compute_indicators(data_3d, connectivity=1,
                                           unit_component_size=4,
                                           normalization='normal')
    indicators_list_3 = compute_indicators(data_3d, connectivity=3,
                                           unit_component_size=2,
                                           normalization='linear')
    expected_keys = {'prop', 'proba', 'density', 'unit_prop',
                     'traversing_prop', 'num_connected', 'box_ratio',
                     'face_cell_ratio', 'sphericity', 'adj_to_0_prop'}
    for i, indicators_list in enumerate([indicators_list_1,
                                         indicators_list_2,
                                         indicators_list_3]):
        for indicators in indicators_list:
            assert indicators.keys() >= expected_keys, (
                f'Missing keys for list {i}')
            for ind_name, values in indicators.items():
                assert isinstance(values, list)
                assert len(values) == 5, (
                    f'Wrong shape for indicator {ind_name}, list {i}')
    # Case wrong data type
    data_wrong = data_2d.astype(np.uint16)
    with pytest.raises(ValueError, match='.*uint16.*'):
        compute_indicators(data_wrong)
    # Case wrong data dim
    data_wrong = np.expand_dims(data_3d, axis=0)
    with pytest.raises(ValueError, match=f'.*{data_wrong.ndim}.*'):
        compute_indicators(data_wrong)
    # Case wrong connectivity
    with pytest.raises(ValueError, match='.*5.*'):
        compute_indicators(data_2d, connectivity=5)
    # Case connectivity == 3 in 2D
    with pytest.raises(ValueError, match='Connectivity 3 is not supported '
                       'for 2D images.'):
        compute_indicators(data_2d, connectivity=3)
    # Case wrong normalization
    with pytest.raises(ValueError, match='.*_UNKNOWN_.*'):
        compute_indicators(data_2d, normalization='_UNKNOWN_')
