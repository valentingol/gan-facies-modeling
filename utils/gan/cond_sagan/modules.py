# Code adapted from https://github.com/heykeetae/Self-Attention-GAN
"""Modules for SAGAN model."""

from typing import List, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import nn

from utils.configs import ConfigType
from utils.data.process import color_data_np
from utils.gan.attention import SelfAttention, TensorAndAttn
from utils.gan.spectral import SpectralNorm


class CondSAGenerator(nn.Module):
    """Self-attention generator."""

    def __init__(self, n_classes: int, model_config: ConfigType) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.data_size = model_config.data_size
        cond_dim_ratio = int(model_config.cond_dim_ratio)

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
                base_dim = model_config.g_conv_dim * self.data_size // 8
                block = self._make_gen_block(model_config.z_dim, base_dim,
                                             kernel_size=4, stride=1,
                                             padding=0)
                cond_dim = base_dim // cond_dim_ratio
                cond_block = self._make_cond_block(out_size=2 * 2**i,
                                                   cond_dim=cond_dim)
                current_dim = base_dim + cond_dim
            else:
                base_dim = base_dim // 2
                block = self._make_gen_block(current_dim, base_dim,
                                             kernel_size=4, stride=2,
                                             padding=1)
                cond_dim = base_dim // cond_dim_ratio
                cond_block = self._make_cond_block(out_size=2 * 2**i,
                                                   cond_dim=cond_dim)
                current_dim = base_dim + cond_dim
            # Add conv blocks to the model
            setattr(self, f'conv{i}', block)
            setattr(self, f'conv_cond{i}', cond_block)

            if make_attention[i - 1]:
                attn = SelfAttention(current_dim,
                                     full_values=model_config.full_values)
                # Add self-attention to the model
                setattr(self, f'attn{attn_id}', attn)
                attn_id += 1

        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(current_dim, n_classes, kernel_size=4, stride=2,
                               padding=1))

        self.init_weights(model_config.init_method)

    def init_weights(self, init_method: str) -> None:
        """Initialize weights."""
        if init_method == 'default':
            return
        for _, param in self.named_parameters():
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

    def _make_cond_block(self, out_size: int, cond_dim: int) -> nn.Module:
        """Return conditional part block."""
        class CondModule(nn.Module):
            """Module for conditional part."""

            def __init__(self, in_size: int, in_dim: int, out_size: int,
                         out_dim: int) -> None:
                super().__init__()
                self.in_size = in_size
                self.out_size = out_size
                self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)

            def forward(self, pixel_maps: torch.Tensor) -> torch.Tensor:
                """Forward pass: mixed pooling then conv 1*1."""
                pixel_indic = pixel_maps[:, 0:1]
                pixel_cls = pixel_maps[:, 1:]
                kernel_size = self.in_size // self.out_size
                resized_indic = nn.MaxPool2d(kernel_size=kernel_size,
                                             stride=kernel_size)(pixel_indic)
                resized_cls = nn.AvgPool2d(kernel_size=kernel_size,
                                           stride=kernel_size)(pixel_cls)
                resized_maps = torch.cat([resized_indic, resized_cls], dim=1)
                out = self.conv(resized_maps)
                return out

        return CondModule(self.data_size, self.n_classes, out_size, cond_dim)

    def forward(self, z: torch.Tensor,
                pixel_maps: torch.Tensor,
                with_attn: bool = False) -> TensorAndAttn:
        """Forward pass.

        Parameters
        ----------
        z : torch.Tensor
            Random input of shape (B, z_dim)
        pixel_maps : torch.Tensor
            Binary pixel maps of shape (B, num_classes, data_size, data_size).
            First channel is pixel sampled indicator. The rest defines
            the class of sampled pixel.
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
            x_cond = getattr(self, f'conv_cond{i}')(pixel_maps)
            x = torch.cat([x, x_cond], dim=1)
            if self.make_attention[i - 1]:
                x, att = getattr(self, f'attn{len(att_list) + 1}')(x)
                att_list.append(att)
        # Here x is of shape (B, g_conv_dim, H/2, W/2)
        x = self.conv_last(x)  # shape (B, n_classes, H, W)
        x = nn.Softmax(dim=1)(x)
        if with_attn:
            return x, att_list
        return x

    def generate(self, z_input: torch.Tensor, pixel_maps: torch.Tensor,
                 with_attn: bool = False) -> Tuple[np.ndarray,
                                                   List[torch.Tensor]]:
        """Return generated images and eventually attention list."""
        out, attn_list = self.forward(z_input, pixel_maps, with_attn=True)
        # Quantize + color generated data
        out = torch.argmax(out, dim=1)
        out = out.detach().cpu().numpy()
        images = color_data_np(out)
        if with_attn:
            return images, attn_list
        return images, []
