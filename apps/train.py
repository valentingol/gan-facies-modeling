"""Train and test the SAGAN model."""

import wandb
from torch.backends import cudnn

from utils.configs import ConfigType, GlobalConfig
from utils.data.data_loader import DataLoader2DFacies
from utils.sagan.trainer import TrainerSAGAN


def train(config: ConfigType) -> None:
    """Train and test the SAGAN model."""
    # For fast training
    cudnn.benchmark = True

    if config.train:
        batch_size = config.training.batch_size
    else:
        # TODO: add test batch size
        raise NotImplementedError('Test not implemented yet! (TODO)')

    # Data loader
    data_loader = DataLoader2DFacies(dataset_path=config.dataset_path,
                                     data_size=config.model.data_size,
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=config.num_workers).loader()

    if config.train:
        architecture = config.model.architecture
        if architecture == 'sagan':
            trainer = TrainerSAGAN(data_loader, config)
        else:
            raise NotImplementedError(f'Architecture "{architecture}" '
                                      'is not implemented!')
        trainer.train()

    # TODO: test the model here


def run() -> None:
    """Run the train using global config."""
    train(global_config)


def run_wandb() -> None:
    """Run the train using wandb."""
    wandb.init(
        config=global_config.get_dict(),
        entity=global_config.wandb.entity,
        project=global_config.wandb.project,
        mode=global_config.wandb.mode,
    )
    if global_config.wandb.sweep is None:
        train(global_config)
    else:
        # Update config for sweep (if any)
        config = GlobalConfig.load_config(wandb.config,
                                          do_not_merge_command_line=True)
        train(config)


if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='configs/exp/base.yaml')
    if global_config.wandb.use_wandb:
        if global_config.wandb.sweep is not None:
            sweep_id = wandb.sweep(
                sweep=global_config.wandb.sweep,
                entity=global_config.wandb.entity,
                project=global_config.wandb.project,
            )
            wandb.agent(sweep_id, function=run_wandb)
        else:
            run_wandb()
    else:
        run()
