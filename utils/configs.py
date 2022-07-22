"""Global configuration file for the project."""

from typing import Union

import wandb
from rr.ml.config import Configuration

ConfigType = Union[wandb.Config, Configuration]


class GlobalConfig(Configuration):
    """Global configuration file for the project."""

    @staticmethod
    def get_default_config_path() -> str:
        """Get the default configuration path."""
        return 'configs/default/main.yaml'

    def parameters_pre_processing(self) -> dict:
        """Pre-processing parameters."""
        return {
            '*_config_path': self.register_as_additional_config_file,
            'config_save_path': self.register_as_experiment_path
        }
