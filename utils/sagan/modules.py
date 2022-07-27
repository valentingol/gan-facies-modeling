# Code adapted from https://github.com/heykeetae/Self-Attention-GAN

"""Modules for SAGAN model."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from utils.data.process import color_data_np
from utils.sagan.spectral import SpectralNorm

TensorWithAttn = Tuple[torch.Tensor, List[torch.Tensor]]


class SelfAttention(nn.Module):
    """Self attention Layer.

    Parameters
    ----------
    in_dim : int
        Input feature map dimension (channels).
    att_dim : int, optional
        Attention map dimension for each query and key
        (and value if full_values is False). By default, in_dim // 8.
    full_values : bool, optional
        Whether to have value dimension equal to full dimension (in_dim)
        or reduced to in_dim // 2. In the latter case, the output of the
        attention is projected to full dimension by an additional
        1*1 convolution. By default, True.
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

    def forward(self, x: torch.Tensor) -> TensorWithAttn:
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
        batch_size, _, width, height = x.size()
        queries = self.query_conv(x).view(batch_size, -1, width * height)
        queries = queries.permute(0, 2, 1)  # (B, W*H~query, Cqk)
        keys = self.key_conv(x).view(batch_size, -1,
                                     width * height)  # (B, Cqk, W*H~key)
        unnorm_attention = torch.bmm(queries, keys)  # (B, W*H~query, W*H~key)
        attention = nn.Softmax(dim=-1)(unnorm_attention)
        values = self.value_conv(x).view(batch_size, -1,
                                         width * height)  # (B, Cv, W*H~value)

        out = torch.bmm(values, attention.permute(0, 2, 1))  # (B, Cv, W*H)
        out = out.view(batch_size, -1, width, height)  # (B, Cv, W, H)

        if self.out_conv is not None:
            out = self.out_conv(out)

        out = self.gamma * out + x

        return out, attention


class SADiscriminator(nn.Module):
    """Self-attention discriminator."""

    def __init__(self, n_classes: int, data_size: int = 64,
                 conv_dim: int = 64, full_values: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.data_size = data_size

        self.conv1 = self._make_disc_block(n_classes, conv_dim, kernel_size=4,
                                           stride=2, padding=1)

        current_dim = conv_dim
        self.conv2 = self._make_disc_block(current_dim, current_dim * 2,
                                           kernel_size=4, stride=2, padding=1)

        current_dim = current_dim * 2
        self.conv3 = self._make_disc_block(current_dim, current_dim * 2,
                                           kernel_size=4, stride=2, padding=1)

        self.attn1 = SelfAttention(current_dim * 2, full_values=full_values)

        if self.data_size == 64:
            current_dim = current_dim * 2
            self.conv4 = self._make_disc_block(current_dim, current_dim * 2,
                                               kernel_size=4, stride=2,
                                               padding=1)
            self.attn2 = SelfAttention(current_dim * 2,
                                       full_values=full_values)

        current_dim = current_dim * 2
        self.conv_last = nn.Sequential(
            nn.Conv2d(current_dim, 1, kernel_size=4),)

    def _make_disc_block(self, in_channels: int, out_channels: int,
                         kernel_size: int, stride: int,
                         padding: int) -> nn.Module:
        """Return a self-attention discriminator block."""
        layers = []
        layers.append(
            SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding)))
        layers.append(nn.LeakyReLU(0.1))
        module = nn.Sequential(*layers)
        return module

    def forward(self, x: torch.Tensor) -> TensorWithAttn:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (B, num_classes, data_size, data_size).

        Returns
        -------
        out : torch.Tensor
            Prediction of discriminator over batch, of shape (B,).
        att_list : list[torch.Tensor]
            Attention maps from all dot product attentions
            of shape (B, W*H~queries, W*H~keys).
        """
        att_list = []
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out, att1 = self.attn1(out)
        att_list.append(att1)

        if self.data_size == 64:
            out = self.conv4(out)
            out, att2 = self.attn2(out)
            att_list.append(att2)

        out = self.conv_last(out).squeeze()

        if out.ndim == 0:  # when batch size is 1
            out = out.unsqueeze(0)

        return out, att_list


class SAGenerator(nn.Module):
    """Self-attention generator."""

    def __init__(self, n_classes: int, data_size: int = 64, z_dim: int = 128,
                 conv_dim: int = 64, full_values: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.data_size = data_size

        repeat_num = int(np.log2(self.data_size)) - 3
        mult = 2**repeat_num  # 8 if data_size=64, 4 if data_size=32

        self.conv1 = self._make_gen_block(z_dim, conv_dim * mult,
                                          kernel_size=4, stride=1, padding=0)

        current_dim = conv_dim * mult
        self.conv2 = self._make_gen_block(current_dim, current_dim // 2,
                                          kernel_size=4, stride=2, padding=1)

        current_dim = current_dim // 2
        self.conv3 = self._make_gen_block(current_dim, current_dim // 2,
                                          kernel_size=4, stride=2, padding=1)

        self.attn1 = SelfAttention(current_dim // 2, full_values=full_values)

        if self.data_size == 64:
            current_dim = current_dim // 2
            self.conv4 = self._make_gen_block(current_dim, current_dim // 2,
                                              kernel_size=4, stride=2,
                                              padding=1)
            self.attn2 = SelfAttention(current_dim // 2,
                                       full_values=full_values)

        current_dim = current_dim // 2

        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(current_dim, n_classes, kernel_size=4, stride=2,
                               padding=1))

    def _make_gen_block(self, in_channels: int, out_channels: int,
                        kernel_size: int, stride: int,
                        padding: int) -> nn.Module:
        """Return a self-attention generator block."""
        layers = []
        layers.append(
            SpectralNorm(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding)))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> TensorWithAttn:
        """Forward pass.

        Parameters
        ----------
        z : torch.Tensor
            Random input of shape (B, z_dim)

        Returns
        -------
        x : torch.Tensor
            Generated data of shape (B, 3, data_size, data_size).
        att_list : list[torch.Tensor]
            Attention maps from all dot product attentions
            of shape (B, W*H~queries, W*H~keys).
        """
        att_list = []
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.conv1(z)
        x = self.conv2(x)
        x = self.conv3(x)
        x, att1 = self.attn1(x)
        att_list.append(att1)

        if self.data_size == 64:
            x = self.conv4(x)
            x, att2 = self.attn2(x)
            att_list.append(att2)

        x = self.conv_last(x)
        x = nn.Softmax(dim=1)(x)
        return x, att_list

    def generate(self, z_input: torch.Tensor,
                 with_attn: bool = False) -> Tuple[np.ndarray,
                                                   List[torch.Tensor]]:
        """Return generated images and eventually attention list."""
        out, attn_list = self.forward(z_input)
        # Quantize + color generated data
        out = torch.argmax(out, dim=1)
        out = out.detach().cpu().numpy()
        images = color_data_np(out)
        if with_attn:
            return images, attn_list
        return images, []
