# Code adapted from https://github.com/heykeetae/Self-Attention-GAN
"""Training class for SAGAN."""

import os
import os.path as osp
from typing import Dict, List, Tuple

import ignite.distributed as idist
import torch
from einops import rearrange
from torch.autograd import Variable
from torch.nn import Module
from torch.optim import Optimizer

from gan_facies.data.data_loader import DistributedDataLoader
from gan_facies.gan.base_trainer import BaseTrainerGAN, BatchType
from gan_facies.gan.cond_sagan.modules import CondSAGenerator
from gan_facies.gan.uncond_sagan.modules import UncondSADiscriminator
from gan_facies.utils.conditioning import generate_pixel_maps
from gan_facies.utils.configs import ConfigType


class CondTrainerSAGAN(BaseTrainerGAN):
    """Trainer for SAGAN.

    Parameters
    ----------
    data_loader : DistributedDataLoader
        Object returning a torch.utils.data.DataLoader when called with
        data_loader.loader() and containing an attribute n_classes.
        The DataLoader should return batches of one-hot-encoded torch
        tensors of shape (batch_size, n_classes, data_size, data_size).
    config : ConfigType
        Global configuration.
    """

    def __init__(self, data_loader: DistributedDataLoader,
                 config: ConfigType) -> None:
        super().__init__(data_loader, config)
        # Create random fixed pixel maps for evaluation
        self.create_pixel_maps()

    def train_generator(self, gen: Module, g_optimizer: Optimizer,
                        disc: Module, real_batch: BatchType,
                        device: torch.device
                        ) -> Tuple[Module, Optimizer, Dict[str, torch.Tensor]]:
        """Train the generator."""
        adv_loss = self.config.training.adv_loss
        losses = {}
        assert len(real_batch) == 2, "Expected data and conditional data"
        pixel_maps = real_batch[1].to(device)

        if (self.gen_ema is None and self.config.training.g_ema_decay < 1.0
                and self.step >= self.config.training.ema_start_step):
            # Start EMA now
            tmp_gen_path = osp.join(self.model_save_path, 'tmp_gen.pth')
            torch.save(gen.state_dict(), tmp_gen_path)
            self.gen_ema = torch.load(tmp_gen_path, map_location='cpu')

        with torch.cuda.amp.autocast(
                enabled=self.config.training.mixed_precision):
            z = torch.randn(self.batch_size,
                            self.config.model.z_dim).to(device)
            fake_data = gen(z, pixel_maps)

            # Compute loss with fake data
            g_out_fake = disc(fake_data)  # batch x n
            if adv_loss == 'wgan-gp':
                g_loss = -g_out_fake.mean()
            elif adv_loss == 'hinge':
                g_loss = -g_out_fake.mean()

            # Add conditional penalty
            pixel_size = self.config.data.pixel_size_cond
            ground_truth = pixel_maps[:, 1:]  # GT for class 1:n_classes
            first_class = torch.all(ground_truth == 0, dim=1)[:, None]
            first_class = pixel_maps[:, 0:1] * first_class  # GT for class 0
            ground_truth = torch.cat([first_class, ground_truth], dim=1)
            diff = pixel_maps[:, 0:1]*fake_data - ground_truth
            cond_loss = torch.norm(diff, dim=(1, 2, 3), p=2).mean()
            cond_loss /= pixel_size
            cond_loss *= self.config.training.cond_penalty

            g_loss += cond_loss

        g_optimizer.zero_grad()
        if self.gen_grad_scaler is not None:
            self.gen_grad_scaler.scale(g_loss).backward()
            self.gen_grad_scaler.step(g_optimizer)
            self.gen_grad_scaler.update()
        else:
            g_loss.backward()
            g_optimizer.step()

        losses['g_loss'] = g_loss, 'green', 6
        losses['g_cond_loss'] = cond_loss, '#027300', 6

        if self.gen_ema is not None:
            ema_decay = self.config.training.g_ema_decay
            with torch.no_grad():
                # Apply EMA on generator
                for (name_param, new_param) in gen.named_parameters():
                    old_param_d = self.gen_ema[name_param].data.to(device)
                    new_param_d = new_param.data.clone()
                    new_param.data.copy_(ema_decay*old_param_d
                                         + (1.0-ema_decay) * new_param_d)
        return gen, g_optimizer, losses

    def train_discriminator(self, disc: Module, d_optimizer: Optimizer,
                            gen: Module, real_batch: BatchType,
                            device: torch.device
                            ) -> Tuple[Module, Optimizer,
                                       Dict[str, torch.Tensor]]:
        """Train the discriminator."""
        adv_loss = self.config.training.adv_loss
        losses = {}
        assert len(real_batch) == 2, "Expected data and conditional data"
        real_data = real_batch[0].to(device)
        pixel_maps = real_batch[1].to(device)

        if (self.disc_ema is None and self.config.training.d_ema_decay < 1.0
                and self.step >= self.config.training.ema_start_step):
            # Start EMA now
            tmp_disc_path = osp.join(self.model_save_path, 'tmp_disc.pth')
            torch.save(disc.state_dict(), tmp_disc_path)
            self.disc_ema = torch.load(tmp_disc_path, map_location='cpu')

        with torch.cuda.amp.autocast(
                enabled=self.config.training.mixed_precision):
            # Compute loss with real data
            d_out_real = disc(real_data)
            if adv_loss == 'wgan-gp':
                d_loss_real = -torch.mean(d_out_real)
            elif adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # Apply Gumbel softmax
            z = torch.randn(self.batch_size,
                            self.config.model.z_dim).to(device)
            fake_data = gen(z, pixel_maps)
            d_out_fake = disc(fake_data)

            if adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        # Backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        if self.disc_grad_scaler is not None:
            self.disc_grad_scaler.scale(d_loss).backward()
            self.disc_grad_scaler.step(d_optimizer)
            self.disc_grad_scaler.update()
        else:
            d_loss.backward()
            d_optimizer.step()
        losses['d_loss'] = d_loss, '#fa4646', 6
        losses['d_loss_real'] = d_loss_real, '#b84a00', 6

        if adv_loss == 'wgan-gp':
            # Compute gradient penalty
            alpha = torch.rand(self.batch_size, 1, 1, 1).to(device)
            alpha = alpha.expand_as(real_data)
            interpolated = Variable(
                alpha * real_data.data + (1-alpha) * fake_data.data,
                requires_grad=True)
            out = disc(interpolated)

            grad = torch.autograd.grad(
                outputs=out,
                inputs=interpolated,
                grad_outputs=torch.ones(out.size()).to(device),
                retain_graph=True,
                create_graph=True,
                only_inputs=True)[0]

            with torch.cuda.amp.autocast(
                    enabled=self.config.training.mixed_precision):
                grad = rearrange(grad, 'shape0 ... -> shape0 (...)')
                grad_l2norm = torch.sqrt(torch.sum(grad**2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss_gp = self.config.training.lambda_gp * d_loss_gp

                # NOTE this trick allows triggering backward gradient
                # hooks for DataDistributedParallel models. It is discuss
                # here: https://github.com/pytorch/pytorch/issues/47562
                d_loss_gp += 0.0 * out[0]

            # Backward + Optimize
            d_optimizer.zero_grad()
            if self.disc_grad_scaler is not None:
                self.disc_grad_scaler.scale(d_loss_gp).backward()
                self.disc_grad_scaler.step(d_optimizer)
                self.disc_grad_scaler.update()
            else:
                d_loss_gp.backward()
                d_optimizer.step()
            losses['d_loss_gp'] = d_loss_gp, '#e06919', 5

        if self.disc_ema is not None:
            ema_decay = self.config.training.d_ema_decay
            with torch.no_grad():
                # Apply EMA on discriminator
                for (name_param, new_param) in disc.named_parameters():
                    old_param_d = self.disc_ema[name_param].data.to(device)
                    new_param_d = new_param.data.clone()
                    new_param.data.copy_(ema_decay*old_param_d
                                         + (1.0-ema_decay) * new_param_d)

        return disc, d_optimizer, losses

    def build_model_opt(self) -> Tuple[Module, Module, Optimizer, Optimizer]:
        """Build generator, discriminator and the optimizers.

        Create the models from SAGAN architecture. Load the parameters
        from recovered checkpoint if enabled, create the Adam optimizer,
        and distribute the models and optimizers on the good device
        automatically. Note that batch norm are also synchronized
        through all the devices.
        """
        config = self.config
        gen = CondSAGenerator(n_classes=self.n_classes,
                              model_config=self.config.model)
        gen_n_params = sum(p.numel() for p in gen.parameters())
        print(f'Generator num parameters: {gen_n_params / 1e6:.3f}M')
        # Discriminator is unconditional
        disc = UncondSADiscriminator(n_classes=self.n_classes,
                                     model_config=self.config.model)
        disc_n_params = sum(p.numel() for p in disc.parameters())
        print(f'Discriminator num parameters: {disc_n_params / 1e6:.3f}M')

        g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad,
                   gen.parameters()), lr=config.training.g_lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )
        d_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad,
                   disc.parameters()), lr=config.training.d_lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )

        # Eventually load pre-trained parameters
        if config.recover_model_step > 0:
            device = idist.device()
            step = config.recover_model_step
            gen.load_state_dict(
                torch.load(
                    os.path.join(self.model_save_path,
                                 f'generator_step_{step}.pth'),
                    map_location=device))
            disc.load_state_dict(
                torch.load(
                    os.path.join(self.model_save_path,
                                 f'discriminator_step_{step}.pth'),
                    map_location=device))
            print(f'Loaded trained models (step: {step}).')

        # Auto-distribute
        gen = idist.auto_model(gen, sync_bn=True)
        disc = idist.auto_model(disc, sync_bn=True)
        g_optimizer = idist.auto_optim(g_optimizer)
        d_optimizer = idist.auto_optim(d_optimizer)

        return gen, disc, g_optimizer, d_optimizer

    def generate_data(self,
                      gen: Module) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Call generate function from generator."""
        gen.eval()
        data, attentions = gen.generate(self.fixed_z, self.fixed_pixel_maps,
                                        with_attn=True)
        return data, attentions

    def create_pixel_maps(self) -> None:
        """Generate pixel maps for evaluation."""
        n_pixels = self.config.data.n_pixels_cond
        pixel_size = self.config.data.pixel_size_cond
        pixel_cls = self.config.data.pixel_classes_cond
        data_size = self.config.model.data_size
        self.fixed_pixel_maps = generate_pixel_maps(batch_size=self.batch_size,
                                                    n_classes=self.n_classes,
                                                    n_pixels=n_pixels,
                                                    pixel_size=pixel_size,
                                                    pixel_classes=pixel_cls,
                                                    data_size=data_size,
                                                    device="cuda:0")
