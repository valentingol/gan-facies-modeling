"""Tests for utils/data/process.py."""

import numpy as np
import pytest

from utils.data.process import (color_data_np, random_crop_np, resize_np,
                                to_img_grid, to_one_hot_np)


@pytest.fixture
def data_one_hot() -> np.ndarray:
    """Return one hot data."""
    struct = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0]])
    struct = np.expand_dims(struct, -1)
    data = np.zeros((4, 5, 2))
    data = np.where(struct == 0, [1, 0], data)
    data = np.where(struct == 1, [0, 1], data)
    return data


@pytest.fixture
def data_int() -> np.ndarray:
    """Return data of type int."""
    data = np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 2, 0, 0],
                     [0, 0, 0, 3, 0]])
    return data


def test_to_one_hot_np(data_int: np.ndarray) -> None:
    """Test to_one_hot_np."""
    data_one_hot = to_one_hot_np(data_int)
    assert data_one_hot.shape == (4, 5, 4)
    assert np.allclose(data_one_hot[0, 0], [1, 0, 0, 0])
    assert np.allclose(data_one_hot[1, 1], [0, 1, 0, 0])
    assert np.allclose(data_one_hot[2, 2], [0, 0, 1, 0])
    assert np.allclose(data_one_hot[3, 3], [0, 0, 0, 1])


def test_resize_np(data_one_hot: np.ndarray) -> None:
    """Test resize_np."""
    # Case min_dim = y
    data = resize_np(data_one_hot, 8)
    assert data.shape == (8, 10, 2)
    assert np.min(data) >= 0
    assert np.allclose(np.sum(data, axis=-1), 1)
    assert data[0, 3, 0] > data[0, 4, 0]
    # Case min_dim = x
    data = np.transpose(data_one_hot, (1, 0, 2))
    data = resize_np(data, 8)
    assert data.shape == (10, 8, 2)
    assert np.min(data) >= 0
    assert np.allclose(np.sum(data, axis=-1), 1)
    assert data[3, 0, 0] > data[4, 0, 0]


def test_random_crop_np(data_one_hot: np.ndarray) -> None:
    """Test random_crop_np."""
    data = random_crop_np(data_one_hot, 3)
    assert data.shape == (3, 3, 2)
    assert np.logical_or(np.isclose(data, 0), np.isclose(data, 1)).all()


def test_color_data_np(data_int: np.ndarray) -> None:
    """Test color_data_np."""
    # Change data_int to have a lot of classes (21 here)
    data_int_sup = data_int.copy()
    data_int_sup[0, 0] = 20

    color_data = color_data_np(data_int_sup)
    assert color_data.shape == (4, 5, 3)
    assert color_data.dtype == np.uint8
    assert np.min(color_data) >= 0
    assert np.max(color_data) <= 255


def test_to_img_grid() -> None:
    """Test to_img_grid."""
    batched_images = np.random.rand(65, 8, 8, 3)
    batched_images = (batched_images * 255).astype(np.uint8)
    img_grid = to_img_grid(batched_images)
    assert img_grid.shape == (64, 64, 3)
