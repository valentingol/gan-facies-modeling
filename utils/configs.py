"""Global configuration file for the project."""

from typing import Union

from yaecs import Configuration


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


ConfigType = Union[Configuration, GlobalConfig]


def merge_configs(config: ConfigType, new_dict_config: dict) -> ConfigType:
    """Merge a new dict config into the current one."""
    # Apply the merge
    new_config = config.copy()
    new_config.merge(new_dict_config,
                     do_not_pre_process=True,
                     verbose=False)
    return new_config
