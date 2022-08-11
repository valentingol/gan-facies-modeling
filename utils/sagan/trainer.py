# Code adapted from https://github.com/heykeetae/Self-Attention-GAN

"""Training class for SAGAN."""

import copy
import json
import os
import os.path as osp
import time
from typing import Dict, List, Union

try:
    import clearml
except ImportError:
    pass
try:
    import wandb
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.configs import ConfigType
from utils.data.process import to_img_grid
from utils.metrics import compute_indicators, wasserstein_distances
from utils.sagan.modules import SADiscriminator, SAGenerator
from utils.train.time_utils import get_delta_eta


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

        # Attributes that will be overwritten when the train will start:
        self.step = -1
        self.start_time = 0.0
        self.start_step = -1

        self.total_time = config.training.total_time
        self.total_step = config.training.total_step

        # Paths
        run_name = config.run_name
        self.attn_path = osp.join('res', run_name, 'attention')
        self.metrics_save_path = osp.join('res', run_name, 'metrics')
        self.model_save_path = osp.join('res', run_name, 'models')
        self.sample_path = osp.join('res', run_name, 'samples')

        self._console = Console()

        self.build_model()

        # Start with trained model
        if config.recover_model_step > 0:
            self.load_pretrained_model(config.recover_model_step)

        # EMA model if required
        if config.training.g_ema_decay < 1.0:
            self.gen_ema = copy.deepcopy(self.gen)

        # Fixed input for sampling
        if config.trunc_ampl > 0:
            # Truncation trick
            self.fixed_z = torch.fmod(torch.randn(config.training.batch_size,
                                                  config.model.z_dim,
                                                  device='cuda'),
                                      config.trunc_ampl)
        else:
            self.fixed_z = torch.randn(config.training.batch_size,
                                       config.model.z_dim,
                                       device='cuda')

        self.indicators_path = ''  # Will be overwritten during training

    def train_generator(self) -> Dict[str, torch.Tensor]:
        """Train the generator."""
        adv_loss = self.config.training.adv_loss
        losses = {}

        z = torch.randn(self.config.training.batch_size,
                        self.config.model.z_dim).cuda()
        fake_data, _ = self.gen(z)

        # Compute loss with fake data
        g_out_fake, _ = self.disc(fake_data)  # batch x n
        if adv_loss == 'wgan-gp':
            g_loss = -g_out_fake.mean()
        elif adv_loss == 'hinge':
            g_loss = -g_out_fake.mean()

        losses['g_loss'] = g_loss

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        ema_decay = self.config.training.g_ema_decay
        ema_start_step = self.config.training.ema_start_step
        if ema_decay < 1.0 and ema_start_step <= self.step:
            with torch.no_grad():
                # Apply EMA on generator
                for (_, old_param), (_, new_param) in \
                        zip(self.gen_ema.named_parameters(),
                            self.gen.named_parameters()):
                    old_param_d = old_param.data.clone()
                    new_param_d = new_param.data.clone()
                    new_param.data.copy_(ema_decay * old_param_d
                                         + (1.0-ema_decay) * new_param_d)
        return losses

    def train_discriminator(
            self, real_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Train the discriminator."""
        adv_loss = self.config.training.adv_loss
        batch_size = self.config.training.batch_size
        losses = {}

        # Compute loss with real data
        d_out_real, _ = self.disc(real_data)
        if adv_loss == 'wgan-gp':
            d_loss_real = -torch.mean(d_out_real)
        elif adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        losses['d_loss_real'] = d_loss_real

        # Apply Gumbel softmax
        z = torch.randn(batch_size, self.config.model.z_dim).cuda()
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

        losses['d_loss'] = d_loss

        if adv_loss == 'wgan-gp':
            # Compute gradient penalty
            alpha = torch.rand(batch_size, 1, 1, 1).cuda()
            alpha = alpha.expand_as(real_data)
            interpolated = Variable(
                alpha * real_data.data + (1-alpha) * fake_data.data,
                requires_grad=True)
            out, _ = self.disc(interpolated)

            grad = torch.autograd.grad(
                outputs=out, inputs=interpolated,
                grad_outputs=torch.ones(out.size()).cuda(), retain_graph=True,
                create_graph=True, only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad**2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1)**2)

            # Backward + Optimize
            d_loss_gp = self.config.training.lambda_gp * d_loss_gp

            self.reset_grad()
            d_loss_gp.backward()
            self.d_optimizer.step()

            losses['d_loss_gp'] = d_loss_gp

        return losses

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
        os.makedirs(osp.join('res', run_name, 'metrics'), exist_ok=True)
        os.makedirs(osp.join('res', run_name, 'models'), exist_ok=True)
        os.makedirs(osp.join('res', run_name, 'samples'), exist_ok=True)

        # Get indicators from training dataset
        self.compute_train_indicators()

        # Data iterator
        data_iter = iter(self.data_loader)

        # Start with trained model
        self.start_step = (config.recover_model_step
                           + 1 if config.recover_model_step > 0 else 0)

        # Start time
        self.start_time = time.time()

        for step in range(self.start_step, self.total_step):
            self.step = step
            # Train Discriminator
            for _ in range(config.training.d_iters):
                self.disc.train()
                self.gen.train()

                try:
                    real_data = next(data_iter)
                except StopIteration:  # Restart data iterator
                    data_iter = iter(self.data_loader)
                    real_data = next(data_iter)

                assert real_data.shape[0] == config.training.batch_size, (
                    'Batch size should always match the value in '
                    f'configurations. Find {real_data.shape[0]} and '
                    f'{config.training.batch_size}. If you are using a torch '
                    'data loader, you may consider set drop_last=True.')
                real_data = real_data.cuda()

                disc_losses = self.train_discriminator(real_data)

            # Train Generator
            gen_losses = self.train_generator()

            losses = {**disc_losses, **gen_losses}

            # Print out log info
            if (step+1) % config.training.log_step == 0:
                self.log(losses)

            # Sample data
            if (step+1) % config.training.sample_step == 0:
                self.save_sample_and_attention()

            if (step+1) % config.training.model_save_step == 0:
                self.save_models()

            if (step+1) % config.training.metric_step == 0:
                self.compute_metrics()

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
        self.save_models(last=True)
        print('Training finished. Final models saved.')

    def log(self, losses: Dict[str, torch.Tensor]) -> None:
        """Log the training progress."""
        rprint = self._console.print

        def log_row(key: str, value: Union[str, float, int],
                    style: str, size: int = 6) -> None:
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
        while hasattr(self.gen, f'attn{attn_id}'):
            avg_gamma = getattr(self.gen, f'attn{attn_id}').gamma
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
            logs = self.get_log_to_dict(metrics=losses,
                                        avg_gammas=avg_gammas)
            wandb.log(logs)
        if self.config.clearml.use_clearml:
            logs = self.get_log_to_dict(metrics=losses,
                                        avg_gammas=avg_gammas)
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

    def reset_grad(self) -> None:
        """Reset the optimizer gradient buffers."""
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def build_model(self) -> None:
        """Build generator, discriminator and the optimizers."""
        config = self.config
        self.gen = SAGenerator(n_classes=self.n_classes,
                               model_config=self.config.model).cuda()
        self.disc = SADiscriminator(n_classes=self.n_classes,
                                    model_config=self.config.model).cuda()

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

    def compute_train_indicators(self) -> None:
        """Compute indicators from training set if not already exist."""
        # Compute indicators for dataset if not provided or get them
        dataset_body, _ = os.path.splitext(self.config.dataset_path)
        self.indicators_path = dataset_body + '_indicators.json'
        if not osp.exists(self.indicators_path):
            print('Compute indicators from training dataset (used for '
                  'metrics)...')
            data = np.load(self.config.dataset_path)
            np.random.shuffle(data)
            # Reduce the size of the dataset to speed up computation
            if len(data) > 2000:
                data = data[:2000]
            # Compute dataset indicators
            indicators_list = compute_indicators(
                data, **self.config.metrics.get_dict())
            # Save indicators in the same folder as the dataset
            with open(self.indicators_path, 'w', encoding='utf-8') as file_out:
                json.dump(indicators_list, file_out, separators=(',', ':'),
                          sort_keys=False, indent=4)

    def save_sample_and_attention(self) -> None:
        """Save sample images and eventually attention maps."""
        self.gen.eval()
        step = self.step
        images, gen_attns = self.gen.generate(self.fixed_z, with_attn=True)

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
                image=img_grid
            )

        if self.config.save_attn:
            # Save attention
            gen_attns = [attn.detach().cpu().numpy() for attn in gen_attns]
            for i, gen_attn in enumerate(gen_attns):
                attn_path = os.path.join(self.attn_path,
                                         f'gen_attn_step_{step + 1}')
                os.makedirs(attn_path, exist_ok=True)
                np.save(osp.join(attn_path, f'attn_{i}.npy'), gen_attn)

    def save_models(self, last: bool = False) -> None:
        """Save generator and discriminator."""
        if last:
            torch.save(
                self.gen.state_dict(),
                os.path.join(self.model_save_path, 'generator_last.pth'))
            torch.save(
                self.disc.state_dict(),
                os.path.join(self.model_save_path, 'discriminator_last.pth'))
        else:
            step = self.step
            torch.save(
                self.gen.state_dict(),
                os.path.join(self.model_save_path,
                             f'generator_step_{step + 1}.pth'))
            torch.save(
                self.disc.state_dict(),
                os.path.join(self.model_save_path,
                             f'discriminator_step_{step + 1}.pth'))

    def compute_metrics(self) -> None:
        """Compute metrics."""
        print('Computing metrics...')
        config = self.config
        # Generate images (over 500 2D images)
        self.gen.eval()
        print(" -> Generating images", end='\r')
        data_gen = []
        batch_size = config.training.batch_size
        with torch.no_grad():
            for k in range(512 // batch_size):
                if config.trunc_ampl > 0:
                    # Truncation trick
                    z_input = torch.fmod(
                        torch.randn(config.training.batch_size,
                                    config.model.z_dim,
                                    device='cuda'),
                        config.trunc_ampl)
                else:
                    z_input = torch.randn(config.training.batch_size,
                                          config.model.z_dim,
                                          device='cuda')
                out, _ = self.gen(z_input)
                out = torch.argmax(out, dim=1).detach().cpu().numpy()
                data_gen.append(out)
                print(f" -> Generating images: {(k + 1)*batch_size} images",
                      end='\r')
        print()
        data_gen_arr = np.vstack(data_gen)
        data_gen_arr = data_gen_arr.astype(np.uint8)
        print(" -> Computing indicators...")

        # Get reference indicators
        with open(self.indicators_path, 'r', encoding='utf-8') as file_in:
            indicators_list_ref = json.load(file_in)
        if config.training.save_boxes:
            save_boxes_path = osp.join(self.metrics_save_path,
                                       f'boxes_step_{self.step + 1}.png')
        else:
            save_boxes_path = None

        # Compute metrics and save boxes locally if needed
        w_dists: Dict[str, float] = wasserstein_distances(
            data_gen_arr, indicators_list_ref, save_boxes_path=save_boxes_path,
            **config.metrics)[0]

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
                    iteration=self.step + 1
                )

        # Log the metrics in the console and wandb or clearml if enabled
        print("Wasserstein distances to **training** dataset indicators:")
        rprint = self._console.print
        for ind_name, w_dist in w_dists.items():
            rprint(f"{ind_name}:", style='#ddf502', end=' ')
            rprint(f"{w_dist:.3f}", style='bold #ddf502', end=' ')
        if config.wandb.use_wandb:
            wandb.log(w_dists)
        if config.clearml.use_clearml:
            for ind_name, value in w_dists.items():
                clearml.Logger.current_logger().report_scalar(
                    ind_name[:-len('_cls_*')],  # base name of indicator
                    ind_name[-len('cls_*'):],  # class number
                    value=value,
                    iteration=self.step + 1,
                )

        # Save metrics locally in a json file
        save_metrics_path = osp.join(self.metrics_save_path,
                                     f'metrics_step_{self.step + 1}.json')
        with open(save_metrics_path, 'w', encoding='utf-8') as file_out:
            json.dump(w_dists, file_out, separators=(',', ':'),
                      sort_keys=False, indent=4)
