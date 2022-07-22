"""Training class for SAGAN."""

import os
import os.path as osp
import time

import numpy as np
import torch
import wandb
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.configs import ConfigType
from utils.data.process import color_data_np, to_img_grid
from utils.sagan.modules import SADiscriminator, SAGenerator


class TrainerSAGAN():
    """Trainer for SAGAN.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        Pytorch DataLoader returning batches of one-hot-encoded
        torch tensors of shape (batch_size, n_classes,
        data_size, data_size).
    """

    def __init__(self, data_loader: DataLoader, config: ConfigType) -> None:

        # Data loader
        self.data_loader = data_loader
        data = next(iter(self.data_loader))
        self.n_classes = data.shape[1]

        # Config
        self.config = config

        # Path
        run_name = config.run_name
        self.attn_path = osp.join('res', run_name, 'attention')
        self.model_save_path = osp.join('res', run_name, 'models')
        self.sample_path = osp.join('res', run_name, 'samples')

        self.build_model()

        # Start with trained model
        if config.recover_model_step > 0:
            self.load_pretrained_model(config.recover_model_step)

    def train(self) -> None:
        """Train SAGAN."""
        config = self.config
        adv_loss = config.training.adv_loss
        assert adv_loss in ['wgan-gp', 'hinge'], (
            f'Loss "{adv_loss}" is not supported.'
            'Should be "wgan-gp" or "hinge".')

        # Create directories if not exist
        run_name = config.run_name
        os.makedirs(osp.join('res', run_name, 'attention'), exist_ok=True)
        os.makedirs(osp.join('res', run_name, 'models'), exist_ok=True)
        os.makedirs(osp.join('res', run_name, 'samples'), exist_ok=True)

        # Data iterator
        data_iter = iter(self.data_loader)

        # Fixed input for sampling
        fixed_z = torch.randn(config.training.batch_size,
                              config.model.z_dim).cuda()

        # Start with trained model
        if config.recover_model_step:
            start = config.recover_model_step + 1
        else:
            start = 0

        # Start time
        start_time = time.time()

        for step in range(start, config.training.total_step):

            self.disc.train()
            self.gen.train()

            # Train Discriminator
            for _ in range(config.training.d_iters):

                try:
                    real_data = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.data_loader)
                    real_data = next(data_iter)
                real_data = real_data.cuda()
                batch_size = real_data.size(0)

                # Compute loss with real data
                d_out_real, _ = self.disc(real_data)
                if adv_loss == 'wgan-gp':
                    d_loss_real = -torch.mean(d_out_real)
                elif adv_loss == 'hinge':
                    d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

                # Apply Gumbel softmax
                z = torch.randn(batch_size, config.model.z_dim).cuda()
                fake_data, _ = self.gen(z)
                d_out_fake, _ = self.disc(fake_data)

                if adv_loss == 'wgan-gp':
                    d_loss_fake = d_out_fake.mean()
                elif adv_loss == 'hinge':
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                if adv_loss == 'wgan-gp':
                    # Compute gradient penalty
                    alpha = (torch.rand(batch_size, 1, 1,
                                        1).cuda().expand_as(real_data))
                    interpolated = Variable(
                        alpha * real_data.data + (1-alpha) * fake_data.data,
                        requires_grad=True)
                    out, _ = self.disc(interpolated)

                    grad = torch.autograd.grad(
                        outputs=out, inputs=interpolated,
                        grad_outputs=torch.ones(out.size()).cuda(),
                        retain_graph=True, create_graph=True,
                        only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad**2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                    # Backward + Optimize
                    d_loss_gp = config.training.lambda_gp * d_loss_gp

                    self.reset_grad()
                    d_loss_gp.backward()
                    self.d_optimizer.step()

            # Train Generator

            z = torch.randn(batch_size, config.model.z_dim).cuda()
            fake_data, _ = self.gen(z)

            # Compute loss with fake data
            g_out_fake, _ = self.disc(fake_data)  # batch x n
            if adv_loss == 'wgan-gp':
                g_loss = -g_out_fake.mean()
            elif adv_loss == 'hinge':
                g_loss = -g_out_fake.mean()

            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Print out log info
            if (step+1) % config.training.log_step == 0:
                delta_t = int(time.time() - start_time)
                delta_str = (f'{delta_t // 3600:02d}h'
                             f'{(delta_t // 60) % 60:02d}m'
                             f'{delta_t % 60:02d}s')
                eta_t = ((config.training.total_step+start-step-1)
                         * delta_t // (step+1-start))
                eta_str = (f'{eta_t // 3600:02d}h'
                           f'{(eta_t // 60) % 60:02d}m'
                           f'{eta_t % 60:02d}s')
                step_spaces = ' ' * (len(str(config.training.total_step))
                                     - len(str(step + 1)))

                avg_gamma1 = np.abs(self.gen.attn1.gamma.mean().item())

                print(
                    f"Step {step_spaces}{step + 1}/"
                    f"{config.training.total_step} "
                    f"[{delta_str} < {eta_str}] "
                    f"G_loss: {g_loss.item():6.2f} | "
                    f"D_loss: {d_loss.item():6.2f} | "
                    f"D_loss_real: {d_loss_real.item():6.2f} | "
                    f"avg gamma(s): {avg_gamma1:5.3f}", end='')

                if config.model.data_size == 64:
                    avg_gamma2 = np.abs(self.gen.attn2.gamma.mean().item())
                    print(f" - {avg_gamma2:5.3f}")
                else:
                    print()

                if config.wandb.use_wandb:
                    logs = {
                        'sum_losses': g_loss.item() + d_loss.item(),
                        'g_loss': g_loss.item(),
                        'd_loss': d_loss.item(),
                        'd_loss_real': d_loss_real.item(),
                        'avg_gamma1': avg_gamma1
                    }
                    if config.model.data_size == 64:
                        logs['avg_gamma2'] = avg_gamma2
                    wandb.log(logs)

            # Sample data
            if (step+1) % config.training.sample_step == 0:
                fake_data, gen_attns = self.gen(fixed_z)
                # Quantize + color generated data
                fake_data = torch.argmax(fake_data, dim=1)
                fake_data = fake_data.detach().cpu().numpy()
                fake_data = color_data_np(fake_data)

                # Save sample images in a grid
                out_path = os.path.join(self.sample_path,
                                        f"images_step_{step + 1}.png")
                img_grid = to_img_grid(fake_data)
                Image.fromarray(img_grid).save(out_path)

                if config.wandb.use_wandb:
                    images = wandb.Image(img_grid, caption=f"step {step + 1}")
                    wandb.log({"generated_images": images})

                if config.save_attn:
                    # Save attention
                    gen_attns = [
                        attn.detach().cpu().numpy() for attn in gen_attns
                    ]
                    for i, gen_attn in enumerate(gen_attns):
                        np.save(
                            osp.join(self.attn_path,
                                     f'gen_attn{i}_step_{step + 1}.npy'),
                            gen_attn)

            if (step+1) % config.training.model_save_step == 0:
                torch.save(
                    self.gen.state_dict(),
                    os.path.join(self.model_save_path,
                                 f'generator_step_{step + 1}.pth'))
                torch.save(
                    self.disc.state_dict(),
                    os.path.join(self.model_save_path,
                                 f'discriminator_step_{step + 1}.pth'))

            if (config.training.total_time >= 0 and
                    time.time() - start_time >= config.training.total_time):
                print('Maximum time reached (note: total training time is '
                      'set in config.training).')
                break
        print('Training finished.')

    def build_model(self) -> None:
        """Build generator, discriminator and the optimizers."""
        config = self.config
        self.gen = SAGenerator(n_classes=self.n_classes,
                               data_size=config.model.data_size,
                               z_dim=config.model.z_dim,
                               conv_dim=config.model.g_conv_dim).cuda()
        self.disc = SADiscriminator(n_classes=self.n_classes,
                                    data_size=config.model.data_size,
                                    conv_dim=config.model.d_conv_dim).cuda()

        self.g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad,
                   self.gen.parameters()), lr=config.training.g_lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )
        self.d_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad,
                   self.disc.parameters()), lr=config.training.d_lr,
            betas=(config.training.beta1, config.training.beta2),
            weight_decay=config.training.weight_decay,
        )

    def load_pretrained_model(self, step: int) -> None:
        """Load pre-trained model."""
        self.gen.load_state_dict(
            torch.load(
                os.path.join(self.model_save_path,
                             f'generator_step_{step}.pth')))
        self.disc.load_state_dict(
            torch.load(
                os.path.join(self.model_save_path,
                             f'discriminator_step_{step}.pth')))
        print(f'Loaded trained models (step: {step}).')

    def reset_grad(self) -> None:
        """Reset the optimizer gradient buffers."""
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
