"""Tests for utils/data/data_loader.py."""

import os
from typing import Tuple

import numpy as np
import pytest
import pytest_check as check
import torch
from pytest_mock import MockerFixture

from gan_facies.data.data_loader import (DatasetCond2D, DatasetUncond2D,
                                         DistributedDataLoader)
from gan_facies.utils.configs import GlobalConfig


@pytest.fixture
def dataset_path() -> str:
    """Return path to test dataset."""
    return 'tests/datasets/tmp_dataset.npy'


def create_test_dataset(dataset_path: str) -> None:
    """Create test dataset (if not exist)."""
    if not os.path.exists('tests/datasets'):
        os.makedirs('tests/datasets')
    if not os.path.exists(dataset_path):
        dataset = np.random.randint(0, 5, size=(7, 5, 20), dtype=np.uint8)
        dataset[0, 0, 0] = 4  # Ensure that there are 5 classes
        np.save(dataset_path, dataset)


def mock_process(mocker: MockerFixture) -> None:
    """Mock functions from data.process."""
    data_one_hot = np.random.rand(7, 5, 20, 5).astype(np.float32)
    data_resize = np.random.rand(10, 20, 5).astype(np.float32)
    data_crop = np.random.rand(10, 10, 5).astype(np.float32)
    pixels = np.random.rand(10, 10, 5).astype(np.float32)
    # normalize output vectors
    data_resize /= np.sum(data_resize, axis=-1, keepdims=True)
    data_crop /= np.sum(data_crop, axis=-1, keepdims=True)

    mocker.patch('gan_facies.data.process.to_one_hot_np',
                 return_value=data_one_hot)
    mocker.patch('gan_facies.data.process.resize_np',
                 return_value=data_resize)
    mocker.patch('gan_facies.data.process.random_crop_np',
                 return_value=data_crop)
    mocker.patch('gan_facies.data.process.sample_pixels_2d_np',
                 return_value=pixels)


def test_dataset_uncond_2d(dataset_path: str,
                           configs: Tuple[GlobalConfig, GlobalConfig],
                           mocker: MockerFixture) -> None:
    """Test DatasetUncond2D."""
    mock_process(mocker)
    config, _ = configs
    create_test_dataset(dataset_path)
    dataset = DatasetUncond2D(dataset_path=dataset_path, data_size=10,
                              data_config=config.data,
                              augmentation_fn=lambda x: 2 * x)
    check.equal(len(dataset), 7)
    sample = dataset[0]
    check.is_instance(sample, tuple)
    check.equal(len(sample), 1)
    data = sample[0]
    check.is_instance(data, torch.Tensor)
    check.equal(data.size(), (5, 10, 10))
    # Sum of values should be 2 with lambda x: 2 * x augmentation
    print(torch.sum(data, dim=1))
    check.is_true(torch.allclose(torch.sum(data, dim=0),
                                 torch.tensor(2, dtype=torch.float32)))
    os.remove(dataset_path)


def test_dataset_cond_2d(dataset_path: str,
                         configs: Tuple[GlobalConfig, GlobalConfig],
                         mocker: MockerFixture) -> None:
    """Test DatasetCond2D."""
    mock_process(mocker)
    config, _ = configs
    create_test_dataset(dataset_path)
    dataset = DatasetCond2D(dataset_path=dataset_path, data_size=10,
                            data_config=config.data,
                            augmentation_fn=lambda x: 2 * x)
    check.equal(len(dataset), 7)
    sample = dataset[0]
    check.is_instance(sample, tuple)
    check.equal(len(sample), 2)
    data, pixel_maps = sample[0], sample[1]
    check.is_instance(data, torch.Tensor)
    check.is_instance(pixel_maps, torch.Tensor)
    check.equal(data.size(), (5, 10, 10))
    check.equal(data.size(), (5, 10, 10))
    # Sum of values should be 2 with lambda x: 2 * x augmentation
    check.is_true(torch.allclose(torch.sum(data, dim=0),
                                 torch.tensor(2, dtype=torch.float32)))
    os.remove(dataset_path)


def test_distributed_dataloader(dataset_path: str,
                                configs: Tuple[GlobalConfig, GlobalConfig],
                                mocker: MockerFixture) -> None:
    """Test DistributedDataLoader."""
    mock_process(mocker)
    config, _ = configs
    create_test_dataset(dataset_path)
    # Case training = True
    dataloader = DistributedDataLoader(
        dataset_path=dataset_path, data_size=10, training=True,
        data_config=config.data, dataset_class=DatasetUncond2D,
        augmentation_fn=lambda x: 2 * x
    ).loader()

    for sample in dataloader:
        data = sample[0]
        # Sum of values should be 1 without augmentation
        # but 2 with lambda x: 2 * x augmentation
        # Here, data augmentation should be applied
        check.is_true(torch.allclose(torch.sum(data, dim=1),
                                     torch.tensor(2, dtype=torch.float32)))

    # Case training = False
    dataloader = DistributedDataLoader(
        dataset_path=dataset_path, data_size=10, training=False,
        data_config=config.data, dataset_class=DatasetUncond2D,
        augmentation_fn=lambda x: 2 * x
    ).loader()

    for sample in dataloader:
        data = sample[0]
        # Sum of values should be 1 without augmentation
        # but 2 with lambda x: 2 * x augmentation
        # Here, data augmentation should be ignored
        check.is_true(torch.allclose(torch.sum(data, dim=1),
                                     torch.tensor(1, dtype=torch.float32)))

    os.remove(dataset_path)
