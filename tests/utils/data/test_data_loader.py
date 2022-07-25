"""Tests for utils/data/data_loader.py."""

import os

import numpy as np
import pytest
import torch

from utils.data.data_loader import DataLoader2DFacies


@pytest.fixture
def dataset_path() -> str:
    """Return path to test dataset."""
    return 'tests/utils/data/tmp_dataset.npy'


def create_test_dataset(dataset_path: str) -> None:
    """Create test dataset (if not exist)."""
    if not os.path.exists(dataset_path):
        dataset = np.random.randint(0, 5, size=(7, 5, 20), dtype=np.uint8)
        dataset[0, 0, 0] = 4  # Ensure that there are 5 classes
        np.save(dataset_path, dataset)


def test_dataloader_2d_facies(dataset_path: str) -> None:
    """Test DataLoader2DFacies."""
    create_test_dataset(dataset_path)

    dataloader = DataLoader2DFacies(dataset_path=dataset_path, data_size=10,
                                    batch_size=3, shuffle=True,
                                    num_workers=2).loader()

    for X in dataloader:
        assert isinstance(X, torch.Tensor)
        assert X.size() == (3, 5, 10, 10)
        assert torch.min(X) >= 0
        assert torch.allclose(torch.sum(X, dim=1),
                              torch.tensor(1, dtype=torch.float32))

    os.remove(dataset_path)
