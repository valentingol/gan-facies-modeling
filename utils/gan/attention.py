"""Class for self-attention."""

from typing import List, Tuple, Union

import torch
from einops import rearrange
from torch import nn

from utils.configs import ConfigType

TensorAndAttn = Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]


class SelfAttention(nn.Module):
    """Self attention Layer.

    Parameters
    ----------
    in_dim : int
        Input feature map dimension (channels).
    attention_config : ConfigType
        Attention configuration. Contains the following keys:

        n_heads : int
            Number of attention heads.
        out_layer : bool
            Whether to use a linear layer at the end of the attention.
        qk_ratio : int
            Dimension for queries and keys is in_dim // qk_ratio.
        v_ratio : int
            Dimension for values is in_dim // v_ratio.
    """

    def __init__(self, in_dim: int, attention_config: ConfigType) -> None:
        super().__init__()
        self.chanel_in = in_dim
        qk_dim = in_dim // attention_config.qk_ratio
        v_dim = in_dim // attention_config.v_ratio
        self.n_heads = attention_config.n_heads

        if attention_config.v_ratio != 1 and not attention_config.out_layer:
            raise ValueError("Found v_ratio != 1 and out_layer=False in "
                             "config. Please set v_ratio = 1 to match the "
                             "value dimension if you don't want an output "
                             "layer.")

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=qk_dim,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=qk_dim,
                                  kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=v_dim,
                                    kernel_size=1)
        if attention_config.out_layer:
            self.out_conv = nn.Conv2d(in_channels=v_dim, out_channels=in_dim,
                                      kernel_size=1)
        else:
            self.out_conv = None

        # gamma: learned scale factor for connection
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> TensorAndAttn:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature maps of shape (B, C, W, H).

        Returns
        -------
        out : torch.Tensor
            Self attention maps + input feature maps.
        attention : torch.Tensor
            Attention maps of shape (B, W*H~queries, W*H~keys)
        """
        _, _, width, height = x.size()
        # Apply 1*1 convolutions
        queries = self.query_conv(x)
        keys = self.key_conv(x)
        values = self.value_conv(x)

        # Notation for dimensions:
        # B=batch dim, H=heads dim, c=channels dim, w=width, h=height

        # Split channels into multiple heads
        queries = rearrange(queries, 'B (H c_qk) h_q w_q -> B H c_qk h_q w_q',
                            H=self.n_heads)
        keys = rearrange(keys, 'B (H c_qk) h_k w_k -> B H c_qk h_k w_k',
                         H=self.n_heads)
        values = rearrange(values, 'B (H c_v) h_v w_v -> B H c_v h_v w_v',
                           H=self.n_heads)

        # Apply dot-product attention
        queries = rearrange(queries, 'B H c_qk h_q w_q -> B H (h_q w_q) c_qk')
        keys = rearrange(keys, 'B H c_qk h_k w_k -> B H c_qk (h_k w_k)')
        unnorm_attention = queries @ keys  # (B, H, hw_query, hw_key)
        attention = nn.Softmax(dim=-1)(unnorm_attention)
        attention_t = rearrange(attention, 'B H hw_q hw_k -> B H hw_k hw_q')

        values = rearrange(values, 'B H c_v h_v w_v -> B H c_v (h_v w_v)')
        out = values @ attention_t  # (B, H, c_value, hw)
        out = rearrange(out, 'B H c_v (h w) -> B H c_v h w', w=width, h=height)

        # Concatenate all the heads
        out = rearrange(out, 'B H c_v h w -> B (H c_v) h w')

        # Eventually apply output layer
        if self.out_conv is not None:
            out = self.out_conv(out)

        # Connect input and attention maps
        out = self.gamma * out + x

        return out, attention
