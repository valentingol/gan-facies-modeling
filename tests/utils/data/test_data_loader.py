"""Tests for utils/data/data_loader.py."""

import os
import typing

import numpy as np
import pytest
import torch

from utils.configs import Configuration, GlobalConfig
from utils.data.data_loader import DataLoader2DFacies, DataLoaderMultiClass


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


@typing.no_type_check
def test_dataloader_multi_class() -> None:
    """Test DataLoaderMultiClass (abstract class)."""
    # Temporary allow instantiation
    # pylint: disable=abstract-class-instantiated
    DataLoaderMultiClass.__abstractmethods__ = set()

    abstract_dataloadr = DataLoaderMultiClass()
    assert abstract_dataloadr.n_classes == -1
    with pytest.raises(NotImplementedError):
        abstract_dataloadr.loader()


def test_dataloader_2d_facies(dataset_path: str,
                              data_config: Configuration) -> None:
    """Test DataLoader2DFacies."""
    create_test_dataset(dataset_path)
    # Case training = True
    dataloader = DataLoader2DFacies(dataset_path=dataset_path,
                                    data_size=10,
                                    training=True,
                                    data_config=data_config,
                                    augmentation_fn=lambda x: 2 * x).loader()

    for X in dataloader:
        assert isinstance(X, torch.Tensor)
        assert X.size() == (2, 5, 10, 10)  # train batch size 2
        assert torch.min(X) >= 0
        # Sum of values should be 1 without augmentation
        # but 2 with lambda x: 2 * x augmentation
        # Here, data augmentation should be applied
        assert torch.allclose(torch.sum(X, dim=1),
                              torch.tensor(2, dtype=torch.float32))

    # Case training = False

    dataloader = DataLoader2DFacies(dataset_path=dataset_path,
                                    data_size=10,
                                    training=False,
                                    data_config=data_config,
                                    augmentation_fn=lambda x: 2 * x).loader()

    for X in dataloader:
        assert isinstance(X, torch.Tensor)
        assert X.size() == (3, 5, 10, 10)  # test batch size 3
        assert torch.min(X) >= 0
        # Sum of values should be 1 without augmentation
        # but 2 with lambda x: 2 * x augmentation
        # Here, data augmentation should be ignored
        assert torch.allclose(torch.sum(X, dim=1),
                              torch.tensor(1, dtype=torch.float32))

    os.remove(dataset_path)
