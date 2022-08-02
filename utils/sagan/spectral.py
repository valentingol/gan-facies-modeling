# Code adapted from https://github.com/heykeetae/Self-Attention-GAN

"""Spectral normalization class."""

from typing import Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import Parameter


def l2normalize(vect: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply L2 normalization."""
    return vect / (vect.norm() + eps)


class SpectralNorm(nn.Module):
    """Spectral normalization."""

    def __init__(self, module: nn.Module, weight_name: str = 'weight',
                 power_iterations: int = 1) -> None:
        super().__init__()
        self.module = module
        self.weight_name = weight_name
        self.power_iterations = power_iterations
        if not self._are_made_params():
            self._make_params()

    def _update_u_v(self) -> None:
        u_param = getattr(self.module, self.weight_name + "_u")
        v_param = getattr(self.module, self.weight_name + "_v")
        w_param = getattr(self.module, self.weight_name + "_bar")
        w_mat = w_param.view(w_param.data.shape[0], -1).data

        for _ in range(self.power_iterations):
            v_param.data = l2normalize(w_mat.T @ u_param.data)
            u_param.data = l2normalize(w_mat @ v_param.data)
        sigma = u_param.data @ w_mat @ v_param.data
        sigma_param = Parameter(sigma).expand_as(w_param)
        setattr(self.module, self.weight_name, w_param / sigma_param)

    def _are_made_params(self) -> bool:
        try:
            getattr(self.module, self.weight_name + "_u")
            getattr(self.module, self.weight_name + "_v")
            getattr(self.module, self.weight_name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self) -> None:
        w_param = getattr(self.module, self.weight_name)
        height = w_param.data.shape[0]
        width = w_param.view(height, -1).data.shape[1]

        u_param = Parameter(
            w_param.data.new(height).normal_(0, 1), requires_grad=False)
        v_param = Parameter(
            w_param.data.new(width).normal_(0, 1), requires_grad=False)
        u_param.data = l2normalize(u_param.data)
        v_param.data = l2normalize(v_param.data)
        w_bar = Parameter(w_param.data)

        delattr(self.module, self.weight_name)

        self.module.register_parameter(self.weight_name + "_u", u_param)
        self.module.register_parameter(self.weight_name + "_v", v_param)
        self.module.register_parameter(self.weight_name + "_bar", w_bar)

    def forward(self, *module_args: Optional[Tuple],
                **module_kwargs: Optional[Mapping]) -> torch.Tensor:
        """Apply spectral normalization."""
        self._update_u_v()
        return self.module.forward(*module_args, **module_kwargs)
