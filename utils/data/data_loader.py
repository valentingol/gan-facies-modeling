"""Data classes."""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data.process import random_crop_np, resize_np, to_one_hot_np


class DataLoader2DFacies():
    """Data loading class for facies 2D matrices.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset that is a .npy file.
    data_size : int
        Size of the data in the dataset. Data will be resized to
        if the current size is below or randomly cropped if the current
        size is above.
    batch_size : int
        Size of the batch.
    shuffle : bool, optional
        Whether to shuffle the dataset or not. By default, True.
    num_workers : int, optional
        Number of workers to use for the data loader.
        If 0, no parallelism is apply.By default, 0.
    """

    def __init__(self, dataset_path: str, data_size: int, batch_size: int,
                 shuffle: bool = True, num_workers: int = 0) -> None:
        self.dataset = Dataset2DFacies(dataset_path, data_size)
        self.n_classes = self.dataset.n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def loader(self) -> torch.utils.data.DataLoader:
        """Return the data loader (Pytorch DataLoader).

        Returns
        -------
        loader : torch.utils.data.DataLoader
            Pytorch DataLoader returning batches of one-hot-encoded
            torch tensors of shape (batch_size, n_classes,
            data_size, data_size).
        """
        loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                             batch_size=self.batch_size,
                                             shuffle=self.shuffle,
                                             num_workers=self.num_workers,
                                             drop_last=True)
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

    __getitem__
        Return a float32 torch tensor (CPU) of shape
        (n_classes, data_size, data_size)
    """

    def __init__(self, dataset_path: str, data_size: int) -> None:
        self.data_size = data_size
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

        # Chanel first
        sample = np.transpose(sample, (2, 0, 1))
        return torch.from_numpy(sample).type(torch.float32)
