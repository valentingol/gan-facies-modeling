# Code adapted from https://github.com/heykeetae/Self-Attention-GAN
"""Modules for SAGAN model."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import nn

import gan_facies.data.process as proc
import gan_facies.gan.attention as att
import gan_facies.gan.initialization as init
import gan_facies.gan.spectral as spec
from gan_facies.utils.configs import ConfigType


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
                with_attn: bool = False) -> att.TensorAndAttn:
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
        attn_list: List[torch.Tensor] = []
        x = rearrange(z, '(B h w) z_dim -> B z_dim h w', h=1, w=1)
        for i in range(1, self.num_blocks):
            x = getattr(self, f'conv{i}')(x)
            x_cond = getattr(self, f'conv_cond{i}')(pixel_maps)
            x = torch.cat([x, x_cond], dim=1)
            if self.make_attention[i - 1]:
                x, attn = getattr(self, f'attn{len(attn_list) + 1}')(x)
                attn_list.append(attn)
        # Here x is of shape (B, g_conv_dim, H/2, W/2)
        x = self.conv_last(x)  # shape (B, n_classes, H, W)
        x = nn.Softmax(dim=1)(x)
        if with_attn:
            return x, attn_list
        return x

    def generate(self, z_input: torch.Tensor, pixel_maps: torch.Tensor,
                 with_attn: bool = False) -> Tuple[np.ndarray,
                                                   List[torch.Tensor]]:
        """Return generated images and eventually attention list."""
        out, attn_list = self.forward(z_input, pixel_maps, with_attn=True)
        # Quantize + color generated data
        out = torch.argmax(out, dim=1)
        out = out.detach().cpu().numpy()
        images = proc.color_data_np(out)
        if with_attn:
            return images, attn_list
        return images, []

    def proba_map(self, z_input: torch.Tensor, pixel_map: torch.Tensor,
                  batch_size: Optional[int] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return probability map (mean and std) from batch of noise.

        Parameters
        ----------
        z_input : torch.Tensor
            Batch of random input noise use for stochastic simulation.
            Should be of shape (num_sample, z_dim).
        pixel_map : torch.Tensor
            Single pixel map used to condition the generator.
            Should be of shape (num_classes, data_size, data_size)
        batch_size : int | None, optional
            Batch size to use for inference. If None, the length
            of z_input is used. By default None.

        Returns
        -------
        proba_mean : np.ndarray
            Mean probability map of shape (data_size, data_size, n_classes).
        proba_std : np.ndarray
            Std probability map of shape (data_size, data_size, n_classes).
        proba_color : np.ndarray
            Colored mean probability map of shape (data_size, data_size, 3).
        """
        batch_size = batch_size or len(z_input)
        pixel_maps = pixel_map.repeat(batch_size, 1, 1, 1)
        out = []
        for i in range(0, len(z_input), batch_size):
            out.append(self.forward(z_input[i:i + batch_size], pixel_maps))
        # out_tensor shape: (num_sample, num_classes, h, w)
        out_tensor = torch.cat(out, dim=0)
        # proba_mean/std shape: (num_classes, h, w)
        proba_mean = torch.mean(out_tensor, dim=0)
        proba_std = torch.std(out_tensor, dim=0)
        proba_mean = rearrange(proba_mean, 'c h w -> h w c')
        proba_std = rearrange(proba_std, 'c h w -> h w c')
        proba_mean = proba_mean.detach().cpu().numpy()
        proba_std = proba_std.detach().cpu().numpy()
        proba_color = proc.continuous_color_data_np(proba_mean)
        return proba_mean, proba_std, proba_color
