"""Tests for utils/conditioning.py."""

import pytest_check as check
import torch

from utils.conditioning import colorize_pixel_map, generate_pixel_maps


def test_generate_pixel_maps() -> None:
    """Test generate_pixel_maps."""
    # Case n_pixels is int
    pixel_maps = generate_pixel_maps(batch_size=2, n_classes=3, n_pixels=5,
                                     data_size=32)
    check.equal(pixel_maps.shape, (2, 3, 32, 32))
    check.equal(pixel_maps.dtype, torch.float32)
    check.is_true((torch.sum(pixel_maps[:, 0], dim=(1, 2)) == 5.0).all())
    check.is_true(((pixel_maps == 0.0) | (pixel_maps == 1.0)).all())
    masked_pixels = (pixel_maps[:, 0:1] == 0.0)
    check.is_true((masked_pixels * pixel_maps == 0.0).all())
    check.is_true(((torch.sum(pixel_maps[:, 1:], dim=1) == 0.0)
                   | (torch.sum(pixel_maps[:, 1:], dim=1) == 1.0)).all())

    # Case n_pixels is list of length 2
    pixel_maps = generate_pixel_maps(batch_size=2, n_classes=3,
                                     n_pixels=[5, 10], data_size=32)
    check.is_true((torch.sum(pixel_maps[:, 0], dim=(1, 2)) >= 5.0).all())
    check.is_true((torch.sum(pixel_maps[:, 0], dim=(1, 2)) <= 10.0).all())

    # Case invalid n_pixels
    with check.raises(ValueError):
        generate_pixel_maps(batch_size=2, n_classes=3, n_pixels=[5, 10, 15],
                            data_size=32)
    with check.raises(ValueError):
        generate_pixel_maps(batch_size=2, n_classes=3,
                            n_pixels='3',  # type: ignore
                            data_size=32)


def test_colorize_pixel_map() -> None:
    """Test colorize_pixel_map."""
    pixel_maps = generate_pixel_maps(batch_size=25, n_classes=3, n_pixels=5,
                                     data_size=32)
    color_pixel_maps = colorize_pixel_map(pixel_maps)
    check.equal(color_pixel_maps.size, (160, 160))
