# Code adapted from https://github.com/heykeetae/Self-Attention-GAN
"""Training class for SAGAN."""

import os
import os.path as osp
import time
from typing import Dict, List, Tuple, Union

try:
    import clearml
except ImportError:
    pass
try:
    import wandb
except ImportError:
    pass

import ignite.distributed as idist
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from rich.console import Console
from torch.autograd import Variable
from torch.nn import Module
from torch.optim import Optimizer

from utils.configs import ConfigType
from utils.data.data_loader import DataLoaderMultiClass
from utils.data.process import to_img_grid
from utils.metrics import compute_save_indicators, evaluate, print_metrics
from utils.sagan.modules import SADiscriminator, SAGenerator
from utils.train.time_utils import get_delta_eta


class TrainerSAGAN():
    """Trainer for SAGAN.

    Parameters
    ----------
    data_loader : DataLoaderMultiClass
        Object returning a torch.utils.data.DataLoader when called with
        data_loader.loader() and containing an attribute n_classes.
        The DataLoader should return batches of one-hot-encoded torch
        tensors of shape (batch_size, n_classes, data_size, data_size).
    config : ConfigType
        Global configuration.
    """

    def __init__(self, data_loader: DataLoaderMultiClass,
                 config: ConfigType) -> None:
        # Data loader
        self.data_loader = data_loader
        self.n_classes = data_loader.n_classes

        # Config
        self.config = config
        config_bs = config.data.train_batch_size
        nproc_per_node = config.distributed.nproc_per_node or 1
        nnodes = config.distributed.nnodes or 1
        self.batch_size = config_bs // (nproc_per_node*nnodes)

        # Attributes that will be overwritten when the train will start:
        self.step = -1
        self.start_time = 0.0
        self.start_step = -1

        self.total_time = config.training.total_time
        self.total_step = config.training.total_step

        # Paths
        run_name = config.run_name
        output_dir = config.output_dir
        self.attn_path = osp.join(output_dir, run_name, 'attention')
        self.metrics_save_path = osp.join(output_dir, run_name, 'metrics')
        self.model_save_path = osp.join(output_dir, run_name, 'models')
        self.sample_path = osp.join(output_dir, run_name, 'samples')

        # EMA models
        # It will be overwritten when EMA actually starts *if enabled*
        self.gen_ema = None
        self.disc_ema = None

        # Fixed input for sampling
        if config.trunc_ampl > 0:
            # Truncation trick
            self.fixed_z = torch.fmod(
                torch.randn(self.batch_size, config.model.z_dim,
                            device='cuda:0'), config.trunc_ampl)
        else:
            self.fixed_z = torch.randn(self.batch_size, config.model.z_dim,
                                       device='cuda:0')

        self.indicators_path = ''  # Will be overwritten during training

        # Gradient scaler for mixed precision training
        if config.training.mixed_precision:
            self.gen_grad_scaler = torch.cuda.amp.GradScaler()
            self.disc_grad_scaler = torch.cuda.amp.GradScaler()
        else:
            self.gen_grad_scaler = None
            self.disc_grad_scaler = None

    def train(self) -> None:
        """Train the GAN on multiple GPUs."""
        config = self.config
        adv_loss = config.training.adv_loss
        assert adv_loss in ['wgan-gp', 'hinge'], (
            f'Loss "{adv_loss}" is not supported.'
            'Should be "wgan-gp" or "hinge".')

        # Create directories if not exist
        run_name = config.run_name
        output_dir = config.output_dir
        os.makedirs(osp.join(output_dir, run_name, 'attention'), exist_ok=True)
        os.makedirs(osp.join(output_dir, run_name, 'metrics'), exist_ok=True)
        os.makedirs(osp.join(output_dir, run_name, 'models'), exist_ok=True)
        os.makedirs(osp.join(output_dir, run_name, 'samples'), exist_ok=True)

        # Get indicators from training dataset
        self.compute_train_indicators()

        # Distribute the training over all available GPUs
        with idist.Parallel(**self.config.distributed) as parallel:
            parallel.run(self.local_train)

        print('Training finished. Final models saved.')

    def local_train(self, rank: int) -> None:
        """Train the model on a single process."""
        config = self.config
        device = idist.device()

        # Create models and optimizers, eventually load parameters from
        # recovered step and distribute the model on the good device
        gen, disc, g_optimizer, d_optimizer = self.build_model_opt()

        # Data iterator
        data_loader = self.data_loader.loader()
        data_iter = iter(data_loader)

        # Start with trained model
        self.start_step = (config.recover_model_step + 1
                           if config.recover_model_step > 0 else 0)

        # Start time
        self.start_time = time.time()

        for step in range(self.start_step, self.total_step):
            self.step = step

            disc.train()
            gen.train()

            # Train Discriminator
            for _ in range(config.training.d_iters):
                try:
                    real_data = next(data_iter)
                except StopIteration:  # Restart data iterator
                    data_iter = iter(data_loader)
                    real_data = next(data_iter)

                assert real_data.shape[0] == self.batch_size, (
                    'Batch size should always match the value in '
                    'configurations divided by world size. Find '
                    f'{real_data.shape[0]} and {self.batch_size}. '
                    'If you are using a torch data loader, you may consider '
                    'set drop_last=True.')
                real_data = real_data.to(device)
                disc, d_optimizer, disc_losses = self.train_discriminator(
                    disc, d_optimizer, gen, real_data, device)

            # Train Generator
            gen, g_optimizer, gen_losses = self.train_generator(
                gen, g_optimizer, disc, device)

            losses = {**disc_losses, **gen_losses}

            if rank == 0:
                base_gen = (gen if not hasattr(gen, 'module')
                            else gen.module)
                base_disc = (disc if not hasattr(disc, 'module')
                             else disc.module)

                # Print out log info
                if (step+1) % config.training.log_step == 0:
                    self.log(losses, base_gen)

                # Sample data
                if (step+1) % config.training.sample_step == 0:
                    self.save_sample_and_attention(base_gen)

                if (step+1) % config.training.model_save_step == 0:
                    self.save_models(base_gen, base_disc, last=False)

                if (step+1) % config.training.metric_step == 0:
                    w_dists = self.compute_metrics(base_gen)
                    self.log_metrics(w_dists)

            if (self.total_time >= 0
                    and time.time() - self.start_time >= self.total_time):
                print('Maximum time reached. Interrupting training '
                      '(note: total training time is '
                      'set in config.training.total_time, set a non-zero '
                      'negative number to disable this feature).')
                break

            if (self.config.training.interrupt_threshold > 0
                    and abs(losses['g_loss'].item()) + abs(losses['d_loss'])
                    >= self.config.training.interrupt_threshold):
                print('Losses are too large. Interrupting training '
                      '(note: the loss threshold is set in '
                      'config.training.interrupt_threshold, set a negative '
                      'number to disable this feature).')
                break
        # Save the final models
        if rank == 0:
            base_gen = gen if not hasattr(gen, 'module') else gen.module
            base_disc = disc if not hasattr(disc, 'module') else disc.module
            self.save_models(base_gen, base_disc, last=True)

    def train_generator(self, gen: Module, g_optimizer: Optimizer,
                        disc: Module, device: torch.device
                        ) -> Tuple[Module, Optimizer, Dict[str, torch.Tensor]]:
        """Train the generator."""
        adv_loss = self.config.training.adv_loss
        losses = {}

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
            fake_data, _ = gen(z)

            # Compute loss with fake data
            g_out_fake, _ = disc(fake_data)  # batch x n
            if adv_loss == 'wgan-gp':
                g_loss = -g_out_fake.mean()
            elif adv_loss == 'hinge':
                g_loss = -g_out_fake.mean()

        g_optimizer.zero_grad()
        if self.gen_grad_scaler is not None:
            self.gen_grad_scaler.scale(g_loss).backward()
            self.gen_grad_scaler.step(g_optimizer)
            self.gen_grad_scaler.update()
        else:
            g_loss.backward()
            g_optimizer.step()

        losses['g_loss'] = g_loss

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
                            gen: Module, real_data: torch.Tensor,
                            device: torch.device
                            ) -> Tuple[Module, Optimizer,
                                       Dict[str, torch.Tensor]]:
        """Train the discriminator."""
        adv_loss = self.config.training.adv_loss
        losses = {}

        if (self.disc_ema is None and self.config.training.d_ema_decay < 1.0
                and self.step >= self.config.training.ema_start_step):
            # Start EMA now
            tmp_disc_path = osp.join(self.model_save_path, 'tmp_disc.pth')
            torch.save(disc.state_dict(), tmp_disc_path)
            self.disc_ema = torch.load(tmp_disc_path, map_location='cpu')

        with torch.cuda.amp.autocast(
                enabled=self.config.training.mixed_precision):
            # Compute loss with real data
            d_out_real, _ = disc(real_data)
            if adv_loss == 'wgan-gp':
                d_loss_real = -torch.mean(d_out_real)
            elif adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # Apply Gumbel softmax
            z = torch.randn(self.batch_size,
                            self.config.model.z_dim).to(device)
            fake_data, _ = gen(z)
            d_out_fake, _ = disc(fake_data)

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
        losses['d_loss_real'] = d_loss_real
        losses['d_loss'] = d_loss

        if adv_loss == 'wgan-gp':
            # Compute gradient penalty
            alpha = torch.rand(self.batch_size, 1, 1, 1).to(device)
            alpha = alpha.expand_as(real_data)
            interpolated = Variable(
                alpha * real_data.data + (1-alpha) * fake_data.data,
                requires_grad=True)
            out, _ = disc(interpolated)

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
            losses['d_loss_gp'] = d_loss_gp

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

    def log(self, losses: Dict[str, torch.Tensor], gen: Module) -> None:
        """Log the training progress."""
        rprint = Console().print

        def log_row(key: str, value: Union[str, float, int], style: str,
                    size: int = 6) -> None:
            """Display a row in the console."""
            value = str(value)[:size].ljust(size)
            rprint(f'{key}: ', style=style, end='')
            rprint(f'{value}', style='bold ' + style, end='', highlight=False)
            print(' | ', end='')

        start_time = self.start_time
        start_step = self.start_step
        step = self.step
        delta_str, eta_str = get_delta_eta(start_time, start_step, step,
                                           self.total_step)
        step_str = f'{step + 1}'.rjust(len(str(self.total_step)))

        avg_gammas, attn_id = [], 1

        while hasattr(gen, f'attn{attn_id}'):
            avg_gamma = getattr(gen, f'attn{attn_id}').gamma
            avg_gamma = torch.abs(avg_gamma).mean().item()
            avg_gammas.append(avg_gamma)
            attn_id += 1
        print(
            f"Step {step_str}/{self.total_step} "
            f"[{delta_str} < {eta_str}] ", end='')
        log_row("G_loss", losses['g_loss'].item(), 'green')
        log_row("D_loss", losses['d_loss'].item(), '#fa4646')
        log_row("D_loss_real", losses['d_loss_real'].item(), '#b84a00')
        if self.config.training.adv_loss == 'wgan-gp':
            log_row("D_loss_gp", losses['d_loss_gp'].item(), '#e06919', size=5)
        rprint("avg gamma(s): ", style='#d670d6', end='', highlight=False)
        for i, avg_gamma in enumerate(avg_gammas):
            if i != 0:
                print(' - ', end='')
            rprint(f"{avg_gamma:5.3f}", end='')
        print()

        if self.config.wandb.use_wandb:
            logs = self.get_log_to_dict(metrics=losses, avg_gammas=avg_gammas)
            wandb.log(logs)
        if self.config.clearml.use_clearml:
            logs = self.get_log_to_dict(metrics=losses, avg_gammas=avg_gammas)
            for name, value in logs.items():
                if name in {'sum_losses', 'abs_losses'}:
                    title = 'Losses properties'
                elif 'loss' in name:
                    title = 'Losses'
                elif 'gamma' in name:
                    title = 'Avg gammas'
                else:
                    title = name
                clearml.Logger.current_logger().report_scalar(
                    title=title,
                    series=name,
                    value=value,
                    iteration=self.step + 1,
                )

    def get_log_to_dict(self, metrics: Dict[str, torch.Tensor],
                        avg_gammas: List[float]) -> Dict[str, float]:
        """Get dict logs from metrics and gammas to dictionary."""
        g_loss, d_loss = metrics['g_loss'].item(), metrics['d_loss'].item()
        logs = {
            'sum_losses': g_loss + d_loss,
            'abs_losses': abs(g_loss) + abs(d_loss)
        }
        for metric_name, metric in metrics.items():
            logs[metric_name] = metric.item()
        for i, avg_gamma in enumerate(avg_gammas):
            logs[f'avg_gamma{i + 1}'] = avg_gamma
        return logs

    def build_model_opt(self) -> Tuple[Module, Module, Optimizer, Optimizer]:
        """Build generator, discriminator and the optimizers.

        Create the models from SAGAN architecture. Load the parameters
        from recovered checkpoint if enabled, create the Adam optimizer,
        and distribute the models and optimizers on the good device
        automatically. Note that batch norm are also synchronized
        through all the devices.
        """
        config = self.config
        device = idist.device()
        gen = SAGenerator(n_classes=self.n_classes,
                          model_config=self.config.model)
        disc = SADiscriminator(n_classes=self.n_classes,
                               model_config=self.config.model)

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

    def compute_train_indicators(self) -> None:
        """Compute indicators from training set if not already exist."""
        # Compute indicators for dataset if not provided or get them
        self.indicators_path = compute_save_indicators(self.data_loader,
                                                       self.config)

    def save_sample_and_attention(self, gen: Module) -> None:
        """Save sample images and eventually attention maps."""
        gen.eval()
        step = self.step
        images, gen_attns = gen.generate(self.fixed_z, with_attn=True)

        # Save sample images in a grid
        out_path = os.path.join(self.sample_path,
                                f"images_step_{step + 1}.png")
        img_grid = to_img_grid(images)
        Image.fromarray(img_grid).save(out_path)

        if self.config.wandb.use_wandb:
            images = wandb.Image(img_grid, caption=f"step {step + 1}")
            wandb.log({"generated_images": images})
        if self.config.clearml.use_clearml:
            clearml.Logger.current_logger().report_image(
                "generated_images",
                "generated_images",
                iteration=step + 1,
                image=img_grid)

        if self.config.save_attn:
            # Save attention
            gen_attns = [attn.detach().cpu().numpy() for attn in gen_attns]
            for i, gen_attn in enumerate(gen_attns):
                attn_path = os.path.join(self.attn_path,
                                         f'gen_attn_step_{step + 1}')
                os.makedirs(attn_path, exist_ok=True)
                np.save(osp.join(attn_path, f'attn_{i}.npy'), gen_attn)

    def save_models(self, gen: Module, disc: Module,
                    last: bool = False) -> None:
        """Save generator and discriminator."""
        if last:
            torch.save(
                gen.state_dict(),
                os.path.join(self.model_save_path, 'generator_last.pth'))
            torch.save(
                disc.state_dict(),
                os.path.join(self.model_save_path, 'discriminator_last.pth'))
        else:
            step = self.step
            torch.save(
                gen.state_dict(),
                os.path.join(self.model_save_path,
                             f'generator_step_{step + 1}.pth'))
            torch.save(
                disc.state_dict(),
                os.path.join(self.model_save_path,
                             f'discriminator_step_{step + 1}.pth'))

    def compute_metrics(self, gen: Module) -> Dict[str, float]:
        """Compute metrics from input generator."""
        print('Computing metrics...')
        w_dists = evaluate(gen=gen, config=self.config, training=True,
                           step=self.step + 1,
                           indicators_path=self.indicators_path,
                           save_json=False, save_csv=True)
        print()
        return w_dists

    def log_metrics(self, w_dists: Dict[str, float]) -> None:
        """Log metrics in console and wandb/clearml if enabled."""
        config = self.config

        if config.training.save_boxes:
            save_boxes_path = osp.join(self.metrics_save_path,
                                       f'boxes_step_{self.step + 1}.png')
        else:
            save_boxes_path = None

        # Log the metrics boxes in wandb or clearml if enabled
        if save_boxes_path:
            fig = plt.gcf()
            if config.wandb.use_wandb:
                wandb.log({"metrics boxes": wandb.Image(fig)})
            if config.clearml.use_clearml:
                clearml.Logger.current_logger().report_matplotlib_figure(
                    'metrics boxes',
                    f'iteration {self.step + 1}',
                    figure=fig,
                    iteration=self.step + 1)

        # Log the metrics in the console and wandb or clearml if enabled
        print("Wasserstein distances to **training** dataset indicators:")
        print_metrics(w_dists, step=self.step + 1)
        if config.wandb.use_wandb:
            wandb.log(w_dists)
        if config.clearml.use_clearml:
            for ind_name, value in w_dists.items():
                if ind_name == 'global':
                    clearml.Logger.current_logger().report_scalar(
                        'global',  # base name of indicator
                        'global',  # class number
                        value=value,
                        iteration=self.step + 1)
                else:
                    clearml.Logger.current_logger().report_scalar(
                        ind_name[:-len('_cls_*')],  # base name of indicator
                        ind_name[-len('cls_*'):],  # class number
                        value=value,
                        iteration=self.step + 1)
