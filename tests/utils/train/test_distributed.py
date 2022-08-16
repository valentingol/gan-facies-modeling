"""Test for utils.train.distributed.py."""

import torch
from torch import nn

from utils.train.distributed import DataParallelModule


def test_data_parallel_module() -> None:
    """Test DataParallelModule class."""
    model = nn.Sequential(nn.Linear(3, 1, bias=False))
    model.n_params = model[0].weight.numel()
    parallel_model = DataParallelModule(model)
    assert parallel_model.n_params == model.n_params
    data = torch.randn(64, 3)
    out = parallel_model(data)
    assert out.shape == (64, 1)
