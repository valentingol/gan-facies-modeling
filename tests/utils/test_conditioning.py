"""Tests for utils/conditioning.py."""

import numpy as np
import pytest_check as check
import torch
from pytest_mock import MockerFixture
from skimage.measure import label

from gan_facies.utils.conditioning import (colorize_pixel_map,
                                           generate_pixel_maps)


def test_generate_pixel_maps() -> None:
    """Test generate_pixel_maps."""
    # Case n_pixels is int, pixel_classes is empty
    pixel_maps = generate_pixel_maps(batch_size=2, n_classes=3, n_pixels=5,
                                     pixel_size=3, pixel_classes=[],
                                     data_size=32)
    check.equal(pixel_maps.shape, (2, 3, 32, 32))
    check.equal(pixel_maps.dtype, torch.float32)
    check.is_true((torch.sum(pixel_maps[:, 0], dim=(1, 2)) == 5 * 3 * 3).all())
    check.is_true(((pixel_maps == 0.0) | (pixel_maps == 1.0)).all())
    pixel_maps_np = pixel_maps.detach().cpu().numpy()
    for i in range(2):
        labels = label(pixel_maps_np[i, 0] + 2*pixel_maps_np[i, 1]
                       + 4*pixel_maps_np[i, 2], background=0, connectivity=1)
        compo_size = [np.count_nonzero(labels == i) >= 9
                      for i in range(1, np.max(labels) + 1)]
        check.is_true(all(compo_size))
    masked_pixels = (pixel_maps[:, 0:1] == 0.0)
    check.is_true((masked_pixels * pixel_maps == 0.0).all())
    check.is_true(((torch.sum(pixel_maps[:, 1:], dim=1) == 0.0)
                   | (torch.sum(pixel_maps[:, 1:], dim=1) == 1.0)).all())

    # Case n_pixels is list of length 2, pixel_classes is not empty
    pixel_maps = generate_pixel_maps(batch_size=2, n_classes=3,
                                     n_pixels=[5, 10], pixel_classes=[0, 2],
                                     pixel_size=1,
                                     data_size=32)
    check.is_true((torch.sum(pixel_maps[:, 0], dim=(1, 2)) >= 5.0).all())
    check.is_true((torch.sum(pixel_maps[:, 0], dim=(1, 2)) <= 10.0).all())
    check.is_true((torch.sum(pixel_maps[:, 1], dim=(1, 2)) == 0.0).all())

    # Case invalid n_pixels
    with check.raises(ValueError):
        generate_pixel_maps(batch_size=2, n_classes=3, n_pixels=[5, 10, 15],
                            pixel_size=1, pixel_classes=[], data_size=32)
    with check.raises(ValueError):
        generate_pixel_maps(batch_size=2, n_classes=3,
                            n_pixels='3', pixel_size=1,  # type: ignore
                            pixel_classes=[], data_size=32)
    # Case invalid pixel_classes
    with check.raises(ValueError):
        generate_pixel_maps(batch_size=2, n_classes=3,
                            n_pixels=1, pixel_size=1,
                            pixel_classes='UNKNOWN',  # type: ignore
                            data_size=32)


def test_colorize_pixel_map(mocker: MockerFixture) -> None:
    """Test colorize_pixel_map."""
    mocker.patch('gan_facies.data.process.color_data_np',
                 return_value=np.random.randint(0, 256, (25, 32, 32, 3),
                                                dtype=np.uint8))
    mocker.patch('gan_facies.data.process.to_img_grid',
                 return_value=np.random.randint(0, 256, (34 * 5, 34 * 5, 3),
                                                dtype=np.uint8))
    pixel_maps = generate_pixel_maps(batch_size=25, n_classes=3, n_pixels=5,
                                     pixel_size=3, pixel_classes=[],
                                     data_size=32)
    color_pixel_maps = colorize_pixel_map(pixel_maps)
    check.is_instance(color_pixel_maps, np.ndarray)
    expected_size = 5 * (32 + 2)
    check.equal(color_pixel_maps.shape, (expected_size, expected_size, 3))
