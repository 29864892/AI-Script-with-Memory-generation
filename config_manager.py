"""
    For system configuration related functionalities
"""
import json
import os
from platform import system

import main as AI_System

def load_config(config_path: str) -> dict:
    """
    Load system configuration from a JSON file.

    Args:
        config_path (str): Path to the config JSON file

    Returns:
        dict: Parsed configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        loaded_config = json.load(f)

    # Basic sanity checks
    if "mode" not in loaded_config:
        raise ValueError("Config missing required key: 'mode'")

    if "paths" not in loaded_config or not isinstance(loaded_config["paths"], dict):
        raise ValueError("Config missing or invalid 'paths' section")

    return loaded_config

#For debugging
def get_path(paths: dict, *keys: str) -> str:
    current = paths
    for key in keys:
        current = current[key]
    return current

if __name__ == "__main__":

    d_config = load_config("config/demo_config.json")

    system = AI_System.run(d_config)

    #system = AI_System.run(config)
    #system.run()