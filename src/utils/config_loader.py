import yaml
import os
from pathlib import Path

CONFIG_PATH = Path("config.yaml")

def load_config():
    # Get the directory of config.yaml file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    path = os.path.join(project_root, "config", "config.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg

def get_models(cfg: dict) -> dict:
    """Return dictionary of models defined in config.yaml."""
    return cfg.get("models", {})

def get_api_keys(cfg: dict) -> dict:
    """Return dictionary of API keys defined in config.yaml."""
    return cfg.get("api_keys", {})