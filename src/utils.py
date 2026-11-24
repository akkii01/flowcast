"""Shared utilities for logging, config loading, and deterministic seeding."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def setup_logger(name: str = "FlowCast") -> logging.Logger:
    """Create and configure a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML project configuration safely."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (if available) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_parent_dir(file_path: str) -> None:
    """Ensure parent directory exists before writing an output file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)