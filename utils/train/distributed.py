"""Utilities classes/functions to build distributed models."""

from typing import Any

import torch


class DataParallelModule(torch.nn.DataParallel):
    """Similar to torch.nn.DataParallel but allows using module attributes."""

    def __getattr__(self, name: str) -> Any:
        """Get attribute from self or get from self.module if not found."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
