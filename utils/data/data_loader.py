"""Data classes."""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import ignite.distributed as idist
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.configs import Configuration
from utils.data.process import random_crop_np, resize_np, to_one_hot_np


class DataLoaderMultiClass(ABC):
    """Abstract base class for multi-class data loaders.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset that is a .npy file.
    data_size : int
        Size of the data in the dataset. Data will be resized to
        if the current size is below or randomly cropped if the current
        size is above.
    training : bool
        Whether the data loader is for training or not.
    data_config : Configuration
        Data sub-configuration. Containing:

        shuffle : bool, optional
            Whether to shuffle the dataset or not. By default, True.
        num_workers : int, optional
            Number of workers to use for the data loader.
            If 0, no parallelism is apply.By default, 0.
        train_batch_size : int
            Size of the batch for training.
        test_batch_size : int
            Size of the batch for testing.
    augmentation_fn : Optional[Callable]
        Augmentation function that takes a sample of type numpy array
        (channel last) and returns a numpy array. Applied in training
        mode only. If None, no augmentation is applied.
        By default, None.
    """

    def __init__(self) -> None:
        self.n_classes = -1  # should be set in subclass

    @abstractmethod
    def loader(self) -> torch.utils.data.DataLoader:
        """Return the data loader, automatically distributed.

        Returns
        -------
        loader : torch.utils.data.DataLoader
            Pytorch DataLoader returning batches of one-hot-encoded
            torch tensors of shape (batch_size, n_classes,
            data_size, data_size). Automatically distributed if multiple
            GPUs are available.
        """
        raise NotImplementedError("Should be implemented in a subclass.")


class DataLoader2DFacies(DataLoaderMultiClass):
    """Data loading class for facies 2D matrices."""

    def __init__(self, dataset_path: str,
                 data_size: int,
                 training: bool,
                 data_config: Configuration,
                 augmentation_fn: Optional[Callable] = None) -> None:
        super().__init__()
        if training:
            self.batch_size = data_config.train_batch_size
            self.dataset = Dataset2DFacies(dataset_path, data_size,
                                           augmentation_fn=augmentation_fn)
        else:
            self.batch_size = data_config.test_batch_size
            self.dataset = Dataset2DFacies(dataset_path, data_size,
                                           augmentation_fn=None)
        self.n_classes = self.dataset.n_classes

        self.num_workers = data_config.num_workers
        self.persistent = data_config.persistant_workers
        self.pin_memory = data_config.pin_memory
        self.prefetch_factor = data_config.prefetch_factor
        self.shuffle = data_config.shuffle

    def loader(self) -> torch.utils.data.DataLoader:
        """Return the data loader, automatically distributed."""
        # NOTE: drop_last=True to maintain constant batch size
        loader = idist.auto_dataloader(dataset=self.dataset,
                                       drop_last=True,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=self.shuffle,
                                       persistent_workers=self.persistent,
                                       pin_memory=self.pin_memory,
                                       prefetch_factor=self.prefetch_factor)
        return loader


class Dataset2DFacies(Dataset):
    """Dataset class for facies 2D matrices.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset that is a .npy file.
    data_size : int
        Size of the data in the dataset. Data will be resized to
        if the current size is below or randomly cropped if the current
        size is above.
    augmentation_fn : Optional[Callable]
        Augmentation function that takes a sample of type numpy array
        (channel last) and returns a numpy array. Applied in training
        mode only. If None, no augmentation is applied.
        By default, None.

    __getitem__
        Return a float32 torch tensor (CPU) of shape
        (n_classes, data_size, data_size)
    """

    def __init__(self, dataset_path: str, data_size: int,
                 augmentation_fn: Optional[Callable] = None) -> None:
        self.data_size = data_size
        self.augmentation_fn = augmentation_fn
        data = np.load(dataset_path)
        self.n_classes = np.max(data) + 1
        # One hot encoding
        self.data = to_one_hot_np(data)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item at index idx."""
        sample = self.data[idx]
        if (sample.shape[0] != self.data_size
                or sample.shape[1] != self.data_size):
            # Resize to min_dim = data_size by preserving
            # the aspect ratio and normalization
            sample = resize_np(sample, self.data_size)
            # Random crop to fit the size (data_size, data_size)
            sample = random_crop_np(sample, self.data_size)
        if self.augmentation_fn is not None:
            sample = self.augmentation_fn(sample)
        # Chanel first
        sample = np.transpose(sample, (2, 0, 1))
        return torch.from_numpy(sample).type(torch.float32)
