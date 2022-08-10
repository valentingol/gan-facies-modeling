"""Train the GAN."""

import os.path as osp

try:
    import clearml
except ImportError:
    pass
try:
    import wandb
except ImportError:
    pass

import torch
from torch.backends import cudnn

from utils.configs import ConfigType, GlobalConfig
from utils.data.data_loader import DataLoader2DFacies
from utils.sagan.trainer import TrainerSAGAN
from utils.train.random_utils import set_global_seed


def train(config: ConfigType) -> None:
    """Train and test the SAGAN model."""
    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available and is required for training.')

    batch_size = config.training.batch_size
    architecture = config.model.architecture

    # Improve reproducibility
    set_global_seed(seed=config.seed)

    # For faster training (but reduce reproducibility!)
    cudnn.benchmark = True

    # Data loader
    data_loader = DataLoader2DFacies(dataset_path=config.dataset_path,
                                     data_size=config.model.data_size,
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=config.num_workers).loader()
    # Model
    if architecture == 'sagan':
        trainer = TrainerSAGAN(data_loader, config)
    else:
        raise NotImplementedError(f'Architecture "{architecture}" '
                                  'is not implemented!')
    # Train
    trainer.train()


def train_wandb() -> None:
    """Run the train using WandB."""
    wandb.init(config=global_config.get_dict(),
               entity=global_config.wandb.entity,
               project=global_config.wandb.project,
               mode=global_config.wandb.mode, group=global_config.wandb.group,
               dir='./wandb_metadata',
               id=global_config.wandb.id,
               resume='allow',
               )
    if global_config.wandb.sweep is None:
        # No sweep, run the train with global config
        train(global_config)
    else:
        # Update config with the sweep

        # Force sweep changes to be at the end of the updated config
        # (under format 'sub.config.key': value)
        config_updated = {**global_config.get_dict(), **dict(wandb.config)}
        # Avoid re-initializing sub-configs with preprocess routines
        config_updated = {
            key: val
            for key, val in config_updated.items()
            if not (key.endswith('config_path') or key == 'config_save_path')
        }
        # Apply the merge
        config = GlobalConfig.load_config(config_updated,
                                          do_not_merge_command_line=True,
                                          overwriting_regime='unsafe')
        train(config)


def train_clearml() -> None:
    """Run the train using ClearML."""
    if global_config.clearml.continue_last_task in {None, False}:
        continue_last_task = False
        reuse_last_task_id = False
    else:
        continue_last_task = global_config.clearml.continue_last_task
        reuse_last_task_id = True
    task = clearml.Task.init(
        project_name=global_config.clearml.project_name,
        task_name=global_config.clearml.task_name,
        tags=global_config.clearml.tags,
        continue_last_task=continue_last_task,
        reuse_last_task_id=reuse_last_task_id,
    )
    task.connect(global_config.get_dict())

    train(global_config)


def main() -> None:
    """Run the train using wandb (+sweep) or not."""
    if global_config.wandb.use_wandb:
        if global_config.clearml.use_clearml:
            raise ValueError(
                'Cannot use both wandb and clearml, please set '
                'config.wandb.use_wandb or config.clearml.use_clearml '
                'to False.')
        if global_config.wandb.sweep is not None:  # WandB with sweep
            sweep_id = wandb.sweep(sweep=global_config.wandb.sweep,
                                   entity=global_config.wandb.entity,
                                   project=global_config.wandb.project,
                                   )
            wandb.agent(sweep_id, function=train_wandb)
        else:  # WandB without sweep
            train_wandb()
    elif global_config.clearml.use_clearml:  # ClearML
        train_clearml()
    else:  # Not remote experiment tracking
        train(global_config)


if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='configs/exp/base.yaml')

    global_config.save(osp.join(global_config.config_save_path, 'config'))
    main()
