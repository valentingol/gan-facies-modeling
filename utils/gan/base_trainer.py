# Code adapted from https://github.com/heykeetae/Self-Attention-GAN
"""Training class for SAGAN."""

import os
import os.path as osp
import time
from abc import ABC, abstractmethod
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
from PIL import Image
from rich.console import Console
from torch.nn import Module
from torch.optim import Optimizer

import utils.auxiliaries as aux
import utils.metrics as met
from utils.configs import ConfigType
from utils.data.data_loader import DistributedDataLoader
from utils.data.process import to_img_grid

BatchType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class BaseTrainerGAN(ABC):
    """Abstract class for distributed GAN trainer.

    This abstract class is used to train a classic GAN with generator
    and discriminator on multiple GPUs, allowing mixed precision
    and EMAs (for generator and discriminator). The losses and metrics
    are computed and logged in the console, in local file and eventually
    on Weights & Biases or ClearML. The attention wights are
    automatically log as long as generator contains layers 'attn{i}'
    and the attention maps itself are saved if needed
    (config.save_attn=True) and exist.

    The methods that need to be implemented in concrete classes are
    train_generator, train_discriminator, build_model_opt
    and generate_data.

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

    @abstractmethod
    def train_generator(self, gen: Module, g_optimizer: Optimizer,
                        disc: Module, real_batch: BatchType,
                        device: torch.device
                        ) -> Tuple[Module, Optimizer, Dict[str, torch.Tensor]]:
        """Train the generator.

        Forward and backward pass for the generator using
        self.gen_grad_scaler for mixed precision
        and self.gen_ema for EMA. An example is provided in
        `uncond_sagan/trainer.py`.

        Parameters
        ----------
        gen : Module
            Generator, distributed using torch ignite (DataParallel
            or DistributedDataParallel).
        g_optimizer : Optimizer
            Optimizer for the generator, distributed with torch ignite.
        disc : Module
            Discriminator, distributed using torch ignite (DataParallel
            or DistributedDataParallel).
        real_batch : BatchType
            Batch of real data plus eventual additional data (like
            sparse pixel maps for conditioning).
        device : torch.device
            Device on which the data and model should be located.

        Returns
        -------
        gen : Module
            Updated discriminator.
        g_optimizer : Optimizer
            Updated optimizer for the discriminator.
        losses : Dict[str, Tuple[torch.Tensor]]
            Dictionary containing the name of the losses as keys and
            (value, style, size) as values. Value is the
            torch scalar corresponding to the loss, style is a string
            containing the style of the loss (e.g. 'green' or '#e06919')
            and size is the length of the loss to log in the console.
            See in `self.log` and `rich` module for details.
        """

    @abstractmethod
    def train_discriminator(self, disc: Module, d_optimizer: Optimizer,
                            gen: Module, real_batch: BatchType,
                            device: torch.device
                            ) -> Tuple[Module, Optimizer,
                                       Dict[str, torch.Tensor]]:
        """Train the discriminator.

        Forward and backward pass for the discriminator using
        self.disc_grad_scaler for mixed precision
        and self.disc_ema for EMA. An example is provided in
        `uncond_sagan/trainer.py`.

        Parameters
        ----------
        disc : Module
            Discriminator, distributed using torch ignite (DataParallel
            or DistributedDataParallel).
        d_optimizer : Optimizer
            Optimizer for the discriminator, distributed with
            torch ignite.
        gen : Module
            Generator, distributed using torch ignite (DataParallel
            or DistributedDataParallel).
        real_batch : BatchType
            Batch of real data plus eventual additional data (like
            sparse pixel maps for conditioning).
        device : torch.device
            Device on which the data and model should be located.

        Returns
        -------
        disc : Module
            Updated discriminator.
        d_optimizer : Optimizer
            Updated optimizer for the discriminator.
        losses : Dict[str, Tuple[torch.Tensor]]
            Dictionary containing the name of the losses as keys and
            (value, style, size) as values. Value is the
            torch scalar corresponding to the loss, style is a string
            containing the style of the loss (e.g. 'green' or '#e06919')
            and size is the length of the loss to log in the console.
            See in `self.log` and `rich` module for details.
        """

    @abstractmethod
    def build_model_opt(self) -> Tuple[Module, Module, Optimizer, Optimizer]:
        """Build generator, discriminator and the optimizers.

        Create the models from architectures. Load the parameters
        from recovered checkpoint if enabled, create the optimizer,
        and distribute the models and optimizers with torch ignite.
        An example is provided in `uncond_sagan/trainer.py`.

        Returns
        -------
        gen : Module
            Distributed generator.
        disc : Module
            Distributed discriminator.
        g_optimizer : Optimizer
            Optimizer for the generator.
        d_optimizer : Optimizer
            Optimizer for the discriminator.
        """

    @abstractmethod
    def generate_data(self,
                      gen: Module) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Call 'generate_data' function from generator with good arguments.

        Parameters
        ----------
        gen : Module
            Generator (not distributed).

        Returns
        -------
        data : torch.Tensor
            Generated data.
        attention : List[torch.Tensor]
            Attention maps. Empty list if no attention maps.
        """

    def create_folders(self) -> None:
        """Create folders to save results in if not exist."""
        run_name = self.config.run_name
        output_dir = self.config.output_dir
        if self.config.save_attn:
            os.makedirs(osp.join(output_dir, run_name, 'attention'),
                        exist_ok=True)
        os.makedirs(osp.join(output_dir, run_name, 'metrics'), exist_ok=True)
        os.makedirs(osp.join(output_dir, run_name, 'models'), exist_ok=True)
        os.makedirs(osp.join(output_dir, run_name, 'samples'), exist_ok=True)

    def train(self) -> None:
        """Train the GAN on multiple GPUs."""
        adv_loss = self.config.training.adv_loss
        assert adv_loss in ['wgan-gp', 'hinge'], (
            f'Loss "{adv_loss}" is not supported.'
            'Should be "wgan-gp" or "hinge".')

        self.create_folders()

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
                    real_batch = next(data_iter)
                except StopIteration:  # Restart data iterator
                    data_iter = iter(data_loader)
                    real_batch = next(data_iter)

                assert real_batch[0].shape[0] == self.batch_size, (
                    'Batch size should always match the value in '
                    'configurations divided by world size. Find '
                    f'{real_batch[0].shape[0]} and {self.batch_size}. '
                    'If you are using a torch data loader, you may consider '
                    'set drop_last=True.')

                disc, d_optimizer, disc_losses = self.train_discriminator(
                    disc, d_optimizer, gen, real_batch, device)

            # Train Generator
            gen, g_optimizer, gen_losses = self.train_generator(
                gen, g_optimizer, disc, real_batch, device)

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
                    # NOTE 'attentions' should be [] if not attention provided
                    images, attentions = self.generate_data(base_gen)
                    self.save_sample_and_attention(images, attentions)

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
        delta_str, eta_str = aux.get_delta_eta(start_time, start_step, step,
                                               self.total_step)
        step_str = f'{step + 1}'.rjust(len(str(self.total_step)))

        print(
            f"Step {step_str}/{self.total_step} "
            f"[{delta_str} < {eta_str}] ", end='')
        for key, (value, style, size) in losses.items():
            log_row(key, value.item(), style, size)

        # Get and log attention weights if any
        avg_gammas, attn_id = [], 1
        while hasattr(gen, f'attn{attn_id}'):
            avg_gamma = getattr(gen, f'attn{attn_id}').gamma
            avg_gamma = torch.abs(avg_gamma).mean().item()
            avg_gammas.append(avg_gamma)
            attn_id += 1
        if avg_gammas:
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
        g_loss = metrics['g_loss'][0].item()
        d_loss = metrics['d_loss'][0].item()
        logs = {
            'sum_losses': g_loss + d_loss,
            'abs_losses': abs(g_loss) + abs(d_loss)
        }
        for metric_name, metric in metrics.items():
            logs[metric_name] = metric[0].item()
        for i, avg_gamma in enumerate(avg_gammas):
            logs[f'avg_gamma{i + 1}'] = avg_gamma
        return logs

    def compute_train_indicators(self) -> None:
        """Compute indicators from training set if not already exist."""
        # Compute indicators for dataset if not provided or get them
        self.indicators_path = met.compute_save_indicators(self.data_loader,
                                                           self.config)

    def save_sample_and_attention(self, images: torch.Tensor,
                                  attentions: List[torch.Tensor]) -> None:
        """Save sample images and eventually attention maps."""
        step = self.step

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
            attentions = [attn.detach().cpu().numpy() for attn in attentions]
            for i, gen_attn in enumerate(attentions):
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
        w_dists = met.evaluate(gen=gen, config=self.config, training=True,
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
        met.print_metrics(w_dists, step=self.step + 1)
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
