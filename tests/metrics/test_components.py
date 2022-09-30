"""Test for utils/metrics/components.py."""

from typing import Tuple

import numpy as np
import pytest
import pytest_check as check

from gan_facies.metrics.components import (get_components_properties,
                                           get_neighbors)


@pytest.fixture
def data() -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D and 3D small random data for testing."""
    data_2d = np.random.randint(0, 4, size=(5, 10, 10), dtype=np.uint8)
    data_2d[0] = 0  # no object to test this limit case
    data_3d = np.random.randint(0, 4, size=(5, 5, 5, 5), dtype=np.uint8)
    data_3d[0] = 0  # no object to test this limit case
    return data_2d, data_3d


def test_get_components_properties(data: Tuple[np.ndarray, np.ndarray]
                                   ) -> None:
    """Test get_components_properties."""
    data_2d, data_3d = data
    # Try connectivity 2 and 3
    properties_list_2d, _ = get_components_properties(data_2d, connectivity=2)
    properties_list_3d, _ = get_components_properties(data_3d, connectivity=3)
    expected_keys = {
        'class', 'mask', 'components', 'areas', 'extents', 'perimeters',
        'n_units', 'mask_unit'
    }
    for properties in properties_list_2d:
        check.equal(set(properties.keys()), expected_keys)
    for properties in properties_list_3d:
        check.equal(set(properties.keys()), expected_keys)


def test_get_neighbors(data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test get_neighbors."""
    data_2d, data_3d = data
    neighbors_2d_con_1 = get_neighbors(data_2d, connectivity=1)
    neighbors_2d_con_2 = get_neighbors(data_2d, connectivity=2)
    neighbors_3d_con_1 = get_neighbors(data_3d, connectivity=1)
    neighbors_3d_con_2 = get_neighbors(data_3d, connectivity=2)
    neighbors_3d_con_3 = get_neighbors(data_3d, connectivity=3)

    check.equal(neighbors_2d_con_1.shape, (5, 10, 10, 4))
    check.equal(neighbors_2d_con_2.shape, (5, 10, 10, 8))
    check.equal(neighbors_3d_con_1.shape, (5, 5, 5, 5, 6))
    check.equal(neighbors_3d_con_2.shape, (5, 5, 5, 5, 18))
    check.equal(neighbors_3d_con_3.shape, (5, 5, 5, 5, 26))
