"""Class for self-attention."""

from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn

TensorAndAttn = Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]


class SelfAttention(nn.Module):
    """Self attention Layer.

    Parameters
    ----------
    in_dim : int
        Input feature map dimension (channels).
    att_dim : int, optional
        Attention map dimension for each query and key
        (and value if full_values is False).
        By default, in_dim // 8.
    full_values : bool, optional
        Whether to have value dimension equal to full dimension
        (in_dim) or reduced to in_dim // 2. In the latter case,
        the output of the attention is projected to full dimension
        by an additional 1*1 convolution. By default, True.
    """

    def __init__(self, in_dim: int, att_dim: Optional[int] = None,
                 full_values: bool = True) -> None:
        super().__init__()
        self.chanel_in = in_dim
        # By default, query and key dimensions are input dim / 8.
        att_dim = in_dim // 8 if att_dim is None else att_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=att_dim,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=att_dim,
                                  kernel_size=1)
        if full_values:
            self.value_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=in_dim,
                                        kernel_size=1)
            self.out_conv = None
        else:
            self.value_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=in_dim // 2,
                                        kernel_size=1)
            self.out_conv = nn.Conv2d(in_channels=in_dim // 2,
                                      out_channels=in_dim,
                                      kernel_size=1)
        # gamma: learned scale factor for residual connection
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
        queries = self.query_conv(x)
        keys = self.key_conv(x)
        values = self.value_conv(x)

        queries = rearrange(queries, 'B Cqk Wq Hq -> B (Wq Hq) Cqk')
        keys = rearrange(keys, 'B Cqk Wk Hk -> B Cqk (Wk Hk)')
        unnorm_attention = torch.bmm(queries, keys)  # (B, WH~query, WH~key)
        attention = nn.Softmax(dim=-1)(unnorm_attention)
        attention_t = rearrange(attention, 'B WHq WHk -> B WHk WHq')

        values = rearrange(values, 'B Cv Wv Hv -> B Cv (Wv Hv)')
        out = torch.bmm(values, attention_t)  # (B, Cv, WH)
        out = rearrange(out, 'B Cv (W H) -> B Cv W H', W=width, H=height)

        if self.out_conv is not None:
            out = self.out_conv(out)

        out = self.gamma * out + x

        return out, attention
