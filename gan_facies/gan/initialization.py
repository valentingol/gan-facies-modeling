"""Function for initializing the weights of the modules."""

from torch import nn


def init_weights(module: nn.Module, init_method: str) -> None:
    """Initialize weights."""
    if init_method == 'default':
        return
    for _, param in module.named_parameters():
        if param.ndim == 4:
            if init_method == 'orthogonal':
                nn.init.orthogonal_(param)
            elif init_method == 'glorot':
                nn.init.xavier_uniform_(param)
            elif init_method == 'normal':
                nn.init.normal_(param, 0, 0.02)
            else:
                raise ValueError(
                    f'Unknown init method: {init_method}. Should be one '
                    'of "default", "orthogonal", "glorot", "normal".')
