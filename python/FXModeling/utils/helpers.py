"""
Helper utilities for the FX modeling project.
"""

import pickle
import yaml
import json
from typing import Any
import os


def save_object(obj: Any, path: str):
    """
    Save an object using pickle.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {path}")


def load_object(path: str) -> Any:
    """
    Load an object using pickle.
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {path}")
    return obj


def load_config(path: str) -> dict:
    """
    Load configuration from YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str):
    """
    Save configuration to YAML file.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to {path}")


def format_number(num: float, precision: int = 4) -> str:
    """
    Format a number for display.
    """
    if abs(num) >= 1e6:
        return f"{num:.2e}"
    elif abs(num) >= 1:
        return f"{num:.{precision}f}"
    elif abs(num) >= 0.01:
        return f"{num:.{precision+2}f}"
    else:
        return f"{num:.2e}"