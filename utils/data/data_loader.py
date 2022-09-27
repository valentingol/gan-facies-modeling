"""Data classes."""

from typing import Callable, Optional, Tuple

import ignite.distributed as idist
import numpy as np
import torch
from torch.utils.data import Dataset

import utils.data.process as proc
from utils.configs import Configuration


class DatasetUncond2D(Dataset):
    """Unconditional dataset class returning 2D matrices.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset that is a .npy file.
    data_size : int
        Size of the data in the dataset. Data will be resized to
        if the current size is below or randomly cropped if the current
        size is above.
    data_config : Configuration
        Configuration.
    augmentation_fn : Optional[Callable]
        Augmentation function that takes a sample of type numpy array
        (channel last) and returns a numpy array. Applied in training
        mode only. If None, no augmentation is applied.
        By default, None.

    __getitem__
        Return a tuple containing a float32 torch tensor (CPU) of shape
        (n_classes, data_size, data_size)
    """

    def __init__(self, dataset_path: str, data_size: int,
                 data_config: Configuration,
                 augmentation_fn: Optional[Callable] = None) -> None:
        self.data_config = data_config
        self.data_size = data_size
        self.augmentation_fn = augmentation_fn
        dataset = np.load(dataset_path)
        self.n_classes = np.max(dataset) + 1
        # One hot encoding
        self.dataset = proc.to_one_hot_np(dataset).astype(np.float32)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Get item at index idx."""
        data = self.dataset[idx]
        if (data.shape[0] != self.data_size
                or data.shape[1] != self.data_size):
            # Resize to min_dim = data_size by preserving
            # the aspect ratio and normalization
            data = proc.resize_np(data, self.data_size)
            # Random crop to fit the size (data_size, data_size)
            data = proc.random_crop_np(data, self.data_size)
        if self.augmentation_fn is not None:
            data = self.augmentation_fn(data)
        # Chanel first
        data = np.transpose(data, (2, 0, 1))
        data = torch.from_numpy(data)
        return (data, )


class DatasetCond2D(Dataset):
    """Conditional dataset class returning 2D matrices and sparse pixel map.

    Pixel map is  a binary map containing n_classes values for each pixel.
    The first value is 1 if a log is sampled in the pixel,
    the n_classes-1 last values determine the class of the log.
    By convention, if the class is 0 (background facies), the last
    values are all equal to 0. More details in the GANSim paper:
    https://link.springer.com/article/10.1007/s11004-021-09934-0

    Parameters
    ----------
    dataset_path : str
        Path to the dataset that is a .npy file.
    data_size : int
        Size of the data in the dataset. Data will be resized to
        if the current size is below or randomly cropped if the current
        size is above.
    data_config : Configuration
        Configuration containing:

        n_pixels : int or List[int] (length 2)
            If int, the number of pixels to sample for each 2D matrix.
            If tuple, the number of pixels to sample for each 2D matrix
            will be sampled uniformly between the two values.

    augmentation_fn : Optional[Callable]
        Augmentation function that takes a sample of type numpy array
        (channel last) and returns a numpy array. Applied in training
        mode only. If None, no augmentation is applied.
        By default, None.

    __getitem__
        Return a tuple containing two float32 torch tensor (CPU) of shape
        (n_classes, data_size, data_size)
        For data and pixel map respectively.
    """

    def __init__(self, dataset_path: str, data_size: int,
                 data_config: Configuration,
                 augmentation_fn: Optional[Callable] = None) -> None:
        self.data_config = data_config
        self.data_size = data_size
        self.n_pixels = data_config.n_pixels_cond
        self.pixel_size = data_config.pixel_size_cond
        self.augmentation_fn = augmentation_fn
        dataset = np.load(dataset_path)
        self.n_classes = np.max(dataset) + 1
        # One hot encoding
        self.dataset = proc.to_one_hot_np(dataset).astype(np.float32)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index idx."""
        data = self.dataset[idx]
        if (data.shape[0] != self.data_size
                or data.shape[1] != self.data_size):
            # Resize to min_dim = data_size by preserving
            # the aspect ratio and normalization
            data = proc.resize_np(data, self.data_size)
            # Random crop to fit the size (data_size, data_size)
            data = proc.random_crop_np(data, self.data_size)
        if self.augmentation_fn is not None:
            data = self.augmentation_fn(data)
        pixel_maps_np = proc.sample_pixels_2d_np(data, self.n_pixels,
                                                 self.pixel_size)
        # Chanel first
        data = np.transpose(data, (2, 0, 1))
        pixel_maps_np = np.transpose(pixel_maps_np, (2, 0, 1))
        # To torch tensor
        data = torch.from_numpy(data)
        pixel_maps = torch.from_numpy(pixel_maps_np).type(torch.float32)
        return data, pixel_maps


class DistributedDataLoader():
    """Distributed data loading class.

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
    dataset_class: Dataset, optional
        Dataset class to use to build the dataset.
        By default, unconditional dataset is used.
    augmentation_fn : Optional[Callable]
        Augmentation function that takes a sample of type numpy array
        (channel last) and returns a numpy array. Applied in training
        mode only. If None, no augmentation is applied.
        By default, None.
    """

    def __init__(self, dataset_path: str, data_size: int, training: bool,
                 data_config: Configuration,
                 dataset_class: Dataset = DatasetUncond2D,
                 augmentation_fn: Optional[Callable] = None) -> None:
        super().__init__()
        if training:
            self.batch_size = data_config.train_batch_size
            self.dataset = dataset_class(dataset_path, data_size,
                                         data_config,
                                         augmentation_fn=augmentation_fn)
        else:
            self.batch_size = data_config.test_batch_size
            self.dataset = dataset_class(dataset_path, data_size, data_config,
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
