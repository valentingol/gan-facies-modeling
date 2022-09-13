"""Tests for utils/data/data_loader.py."""

import os

import numpy as np
import pytest
import pytest_check as check
import torch

from utils.configs import Configuration, GlobalConfig
from utils.data.data_loader import (DatasetCond2D, DatasetUncond2D,
                                    DistributedDataLoader)


@pytest.fixture
def dataset_path() -> str:
    """Return path to test dataset."""
    return 'tests/utils/data/tmp_dataset.npy'


@pytest.fixture
def data_config() -> Configuration:
    """Return data sub-config object for testing."""
    return GlobalConfig.build_from_argv(
        fallback='configs/unittest/data32.yaml').data


def create_test_dataset(dataset_path: str) -> None:
    """Create test dataset (if not exist)."""
    if not os.path.exists(dataset_path):
        dataset = np.random.randint(0, 5, size=(7, 5, 20), dtype=np.uint8)
        dataset[0, 0, 0] = 4  # Ensure that there are 5 classes
        np.save(dataset_path, dataset)


def test_dataset_uncond_2d(dataset_path: str,
                           data_config: Configuration) -> None:
    """Test DatasetUncond2D."""
    create_test_dataset(dataset_path)
    dataset = DatasetUncond2D(dataset_path=dataset_path, data_size=10,
                              data_config=data_config,
                              augmentation_fn=lambda x: 2 * x)
    check.equal(len(dataset), 7)
    sample = dataset[0]
    check.is_instance(sample, tuple)
    check.equal(len(sample), 1)
    data = sample[0]
    check.is_instance(data, torch.Tensor)
    check.equal(data.size(), (5, 10, 10))
    check.greater_equal(torch.min(data), 0)
    # Sum of values should be 2 with lambda x: 2 * x augmentation
    print(torch.sum(data, dim=1))
    check.is_true(torch.allclose(torch.sum(data, dim=0),
                                 torch.tensor(2, dtype=torch.float32)))
    os.remove(dataset_path)


def test_dataset_cond_2d(dataset_path: str,
                         data_config: Configuration) -> None:
    """Test DatasetCond2D."""
    create_test_dataset(dataset_path)
    dataset = DatasetCond2D(dataset_path=dataset_path, data_size=10,
                            data_config=data_config,
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
    check.greater_equal(torch.min(data), 0)
    # Sum of values should be 2 with lambda x: 2 * x augmentation
    check.is_true(torch.allclose(torch.sum(data, dim=0),
                                 torch.tensor(2, dtype=torch.float32)))
    os.remove(dataset_path)


def test_distributed_dataloader(dataset_path: str,
                                data_config: Configuration) -> None:
    """Test DistributedDataLoader."""
    create_test_dataset(dataset_path)
    # Case training = True
    dataloader = DistributedDataLoader(
        dataset_path=dataset_path, data_size=10, training=True,
        data_config=data_config, dataset_class=DatasetUncond2D,
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
        data_config=data_config, dataset_class=DatasetUncond2D,
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
