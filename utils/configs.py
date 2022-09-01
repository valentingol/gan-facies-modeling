"""Global configuration file for the project."""

from typing import Union

from rr.ml.config import Configuration


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
    config_updated = {**config.get_dict(deep=True),
                      **new_dict_config}
    # Avoid re-initializing sub-configs with preprocess routines
    config_updated = {
        key: val
        for key, val in config_updated.items()
        if not (key.endswith('config_path') or key == 'config_save_path')
    }
    # Apply the merge
    new_config = GlobalConfig.load_config(config_updated,
                                          do_not_merge_command_line=True,
                                          overwriting_regime='unsafe')
    return new_config
