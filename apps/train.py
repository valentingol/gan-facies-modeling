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


def train_wandb() -> None:
    """Run the train using wandb."""
    wandb.init(
        config=global_config.get_dict(),
        entity=global_config.wandb.entity,
        project=global_config.wandb.project,
        mode=global_config.wandb.mode,
        group=global_config.wandb.group,
        dir='./wandb_metadata',
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
        config_updated = {key: val for key, val in config_updated.items()
                          if not (key.endswith('config_path')
                          or key == 'config_save_path')}
        # Apply the merge
        config = GlobalConfig.load_config(config_updated,
                                          do_not_merge_command_line=True,
                                          overwriting_regime='unsafe')
        train(config)


def main() -> None:
    """Run the train using wandb (+sweep) or not."""
    if global_config.wandb.use_wandb:
        if global_config.wandb.sweep is not None:
            sweep_id = wandb.sweep(
                sweep=global_config.wandb.sweep,
                entity=global_config.wandb.entity,
                project=global_config.wandb.project,
            )
            wandb.agent(sweep_id, function=train_wandb)
        else:
            train_wandb()
    else:
        train(global_config)


if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='configs/exp/base.yaml')
    main()
