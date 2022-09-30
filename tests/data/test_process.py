"""Tests for utils/data/process.py."""

import numpy as np
import pytest
import pytest_check as check
from skimage.measure import label

from gan_facies.data.process import (color_data_np, continuous_color_data_np,
                                     random_crop_np, resize_np,
                                     sample_pixels_2d_np, to_img_grid,
                                     to_one_hot_np)
from tests.conftest import check_allclose


@pytest.fixture
def data_one_hot() -> np.ndarray:
    """Return one hot data."""
    struct = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0]])
    struct = np.expand_dims(struct, -1)
    data = np.zeros((4, 5, 2))
    data = np.where(struct == 0, [1, 0], data)
    data = np.where(struct == 1, [0, 1], data)
    return data


@pytest.fixture
def data_int() -> np.ndarray:
    """Return data of type int."""
    data = np.array([[0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 2, 0, 0],
                     [0, 0, 0, 3, 0]])
    return data


def test_to_one_hot_np(data_int: np.ndarray) -> None:
    """Test to_one_hot_np."""
    data_one_hot = to_one_hot_np(data_int)
    check.equal(data_one_hot.shape, (4, 5, 4))
    check_allclose(data_one_hot[0, 0], [1, 0, 0, 0])
    check_allclose(data_one_hot[1, 1], [0, 1, 0, 0])
    check_allclose(data_one_hot[2, 2], [0, 0, 1, 0])
    check_allclose(data_one_hot[3, 3], [0, 0, 0, 1])


def test_resize_np(data_one_hot: np.ndarray) -> None:
    """Test resize_np."""
    # Case min_dim = y
    data = resize_np(data_one_hot, 8)
    check.equal(data.shape, (8, 10, 2))
    check.greater_equal(np.min(data), 0)
    check_allclose(np.sum(data, axis=-1), 1)
    check.greater(data[0, 3, 0], data[0, 4, 0])
    # Case min_dim = x
    data = np.transpose(data_one_hot, (1, 0, 2))
    data = resize_np(data, 8)
    check.equal(data.shape, (10, 8, 2))
    check.greater_equal(np.min(data), 0)
    check_allclose(np.sum(data, axis=-1), 1)
    check.greater(data[3, 0, 0], data[4, 0, 0])


def test_random_crop_np(data_one_hot: np.ndarray) -> None:
    """Test random_crop_np."""
    data = random_crop_np(data_one_hot, 3)
    check.equal(data.shape, (3, 3, 2))
    check.is_true(np.logical_or(np.isclose(data, 0),
                                np.isclose(data, 1)).all())


def test_color_data_np(data_int: np.ndarray) -> None:
    """Test color_data_np."""
    # Change data_int to have a lot of classes (21 here)
    data_int_sup = data_int.copy()
    data_int_sup[0, 0] = 20

    color_data = color_data_np(data_int_sup)
    check.equal(color_data.shape, (4, 5, 3))
    check.equal(color_data.dtype, np.uint8)
    check.greater_equal(np.min(color_data), 0)
    check.less_equal(np.max(color_data), 255)


def test_continuous_color_data_np(data_one_hot: np.ndarray) -> None:
    """Test continuous_color_data_np."""
    cont_data_on_hot = np.where(data_one_hot == 1, 0.8, 0.2)
    color_data = continuous_color_data_np(cont_data_on_hot)
    check.equal(color_data.shape, (4, 5, 3))
    check.equal(color_data.dtype, np.uint8)
    check.greater_equal(np.min(color_data), 0)
    check.less_equal(np.max(color_data), 255)


def test_to_img_grid() -> None:
    """Test to_img_grid."""
    batched_images = np.random.rand(65, 8, 8, 3)
    batched_images = (batched_images * 255).astype(np.uint8)
    img_grid = to_img_grid(batched_images)
    check.equal(img_grid.shape, (8 * (8+2), 8 * (8+2), 3))


def test_sample_pixels_2d_np(data_one_hot: np.ndarray) -> None:
    """Test sample_pixels_2d_np."""
    # Case n_pixels is int
    pixel_maps = sample_pixels_2d_np(data_one_hot, n_pixels=2, pixel_size=2)
    check.equal(pixel_maps.shape, (4, 5, 2))
    check.equal(pixel_maps.dtype, np.float32)
    check.equal(np.count_nonzero(pixel_maps[..., 0]), 2 * 2 * 2)
    labels = label(pixel_maps[..., 0] + 2*pixel_maps[..., 1], background=0,
                   connectivity=1)
    compo_size = [np.count_nonzero(labels == i) >= 4
                  for i in range(1, np.max(labels) + 1)]
    check.is_true(all(compo_size))
    check.is_true(((pixel_maps == 0.0) | (pixel_maps == 1.0)).all())
    masked_pixels = (pixel_maps[..., 0] == 0.0)[..., None]
    check.is_true((masked_pixels * pixel_maps == 0.0).all())
    check.is_true(((np.sum(pixel_maps[..., 1:], axis=-1) == 0.0)
                   | (np.sum(pixel_maps[..., 1:], axis=-1) == 1.0)).all())

    # Case n_pixels is list of length 2
    pixel_maps = sample_pixels_2d_np(data_one_hot, [5, 10], pixel_size=1)
    check.greater_equal(np.count_nonzero(pixel_maps[..., 0]), 5)
    check.less_equal(np.count_nonzero(pixel_maps[..., 0]), 10)

    # Case wrong n_pixels value or type
    with check.raises(ValueError):
        sample_pixels_2d_np(data_one_hot, n_pixels='5',  # type: ignore
                            pixel_size=1)
    with check.raises(ValueError):
        sample_pixels_2d_np(data_one_hot, n_pixels=[5, 10, 15], pixel_size=1)
