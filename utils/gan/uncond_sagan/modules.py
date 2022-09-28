# Code adapted from https://github.com/heykeetae/Self-Attention-GAN
"""Modules for SAGAN model."""

from typing import List, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import nn

import utils.data.process as proc
import utils.gan.attention as att
import utils.gan.initialization as init
import utils.gan.spectral as spec
from utils.configs import ConfigType


class UncondSADiscriminator(nn.Module):
    """Self-attention discriminator."""

    def __init__(self, n_classes: int, model_config: ConfigType) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.data_size = model_config.data_size

        datasize_to_num_blocks = {32: 4, 64: 5, 128: 6, 256: 7}
        num_blocks = datasize_to_num_blocks[model_config.data_size]
        self.num_blocks = num_blocks
        # make_attention[i] is True if adding self-attention
        # to {num_blocks-i-1}-th block output in order to have symmetric
        # attention structure in generator and discriminator
        make_attention = [
            num_blocks - i - 1 in model_config.attn_layer_num
            for i in range(num_blocks)
        ]
        self.make_attention = make_attention

        attn_id = 1  # Attention layers id
        for i in range(1, num_blocks):
            if i == 1:  # First block:
                block = self._make_disc_block(n_classes,
                                              model_config.d_conv_dim,
                                              kernel_size=4, stride=2,
                                              padding=1)
                current_dim = model_config.d_conv_dim
            else:
                block = self._make_disc_block(current_dim, current_dim * 2,
                                              kernel_size=4, stride=2,
                                              padding=1)
                current_dim = current_dim * 2
            # Add conv block to the model
            setattr(self, f'conv{i}', block)

            if make_attention[i - 1]:
                attn = att.SelfAttention(
                    in_dim=current_dim,
                    attention_config=model_config.attention)
                # Add self-attention to the model
                setattr(self, f'attn{attn_id}', attn)
                attn_id += 1

        self.conv_last = nn.Sequential(
            nn.Conv2d(current_dim, 1, kernel_size=4),)

        init.init_weights(self, model_config.init_method)

    def _make_disc_block(self, in_channels: int, out_channels: int,
                         kernel_size: int, stride: int,
                         padding: int) -> nn.Module:
        """Return a self-attention discriminator block."""
        layers = []
        layers.append(
            spec.SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding)))
        layers.append(nn.LeakyReLU(0.1))
        module = nn.Sequential(*layers)
        return module

    def forward(self, x: torch.Tensor,
                with_attn: bool = False) -> att.TensorAndAttn:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (B, num_classes, data_size, data_size).
        with_attn : bool, optional
            Whether to return attention maps, by default False

        Returns
        -------
        x : torch.Tensor
            Prediction of discriminator over batch, of shape (B,).
        att_list : list[torch.Tensor]
            Attention maps from all dot product attentions
            of shape (B, W*H~queries, W*H~keys).
        """
        att_list: List[torch.Tensor] = []
        for i in range(1, self.num_blocks):
            x = getattr(self, f'conv{i}')(x)
            if self.make_attention[i - 1]:
                x, att = getattr(self, f'attn{len(att_list) + 1}')(x)
                att_list.append(att)
        # Here x is of shape ([B], d_conv_dim/8, 4, 4)
        x = self.conv_last(x).squeeze()  # shape ([B], )

        if x.ndim == 0:  # when batch size is 1
            x = x.unsqueeze(0)
        if with_attn:
            return x, att_list
        return x


class UncondSAGenerator(nn.Module):
    """Self-attention generator."""

    def __init__(self, n_classes: int, model_config: ConfigType) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.data_size = model_config.data_size

        datasize_to_num_blocks = {32: 4, 64: 5, 128: 6, 256: 7}
        num_blocks = datasize_to_num_blocks[model_config.data_size]
        self.num_blocks = num_blocks
        # make_attention[i] is True if adding self-attention
        # to {i+1}-th block output
        make_attention = [
            i + 1 in model_config.attn_layer_num for i in range(num_blocks)
        ]
        self.make_attention = make_attention

        attn_id = 1  # Attention layers id
        for i in range(1, num_blocks):
            if i == 1:  # First block:
                current_dim = model_config.g_conv_dim * self.data_size // 8
                block = self._make_gen_block(model_config.z_dim, current_dim,
                                             kernel_size=4, stride=1,
                                             padding=0)
            else:
                block = self._make_gen_block(current_dim, current_dim // 2,
                                             kernel_size=4, stride=2,
                                             padding=1)
                current_dim = current_dim // 2
            # Add conv block to the model
            setattr(self, f'conv{i}', block)

            if make_attention[i - 1]:
                attn = att.SelfAttention(
                    in_dim=current_dim,
                    attention_config=model_config.attention)
                # Add self-attention to the model
                setattr(self, f'attn{attn_id}', attn)
                attn_id += 1

        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(current_dim, n_classes, kernel_size=4, stride=2,
                               padding=1))

        init.init_weights(self, model_config.init_method)

    def _make_gen_block(self, in_channels: int, out_channels: int,
                        kernel_size: int, stride: int,
                        padding: int) -> nn.Module:
        """Return a self-attention generator block."""
        layers = []
        layers.append(
            spec.SpectralNorm(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding)))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor,
                with_attn: bool = False) -> att.TensorAndAttn:
        """Forward pass.

        Parameters
        ----------
        z : torch.Tensor
            Random input of shape (B, z_dim)
        with_attn : bool, optional
            Whether to return attention maps, by default False

        Returns
        -------
        x : torch.Tensor
            Generated data of shape (B, 3, data_size, data_size).
        att_list : list[torch.Tensor]
            Attention maps from all dot product attentions
            of shape (B, W*H~queries, W*H~keys).
        """
        att_list: List[torch.Tensor] = []
        x = rearrange(z, '(B h w) z_dim -> B z_dim h w', h=1, w=1)
        for i in range(1, self.num_blocks):
            x = getattr(self, f'conv{i}')(x)
            if self.make_attention[i - 1]:
                x, att = getattr(self, f'attn{len(att_list) + 1}')(x)
                att_list.append(att)
        # Here x is of shape (B, g_conv_dim, H/2, W/2)
        x = self.conv_last(x)  # shape (B, n_classes, H, W)
        x = nn.Softmax(dim=1)(x)
        if with_attn:
            return x, att_list
        return x

    def generate(self, z_input: torch.Tensor, with_attn: bool = False
                 ) -> Tuple[np.ndarray, List[torch.Tensor]]:
        """Return generated images and eventually attention list."""
        out, attn_list = self.forward(z_input, with_attn=True)
        # Quantize + color generated data
        out = torch.argmax(out, dim=1)
        out = out.detach().cpu().numpy()
        images = proc.color_data_np(out)
        if with_attn:
            return images, attn_list
        return images, []
