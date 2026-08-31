"""
config.py

Utility functions for loading YAML configuration files.
"""

from pathlib import Path
import yaml


def load_config(config_path: str):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config.

    Returns:
        dict: Configuration dictionary.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: dict, save_path: str):
    """
    Save configuration dictionary to YAML.

    Args:
        config (dict): Configuration dictionary.
        save_path (str): Output path.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
        )