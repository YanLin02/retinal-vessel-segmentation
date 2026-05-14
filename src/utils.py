from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        if yaml is not None:
            config = yaml.safe_load(f)
        else:
            config = _load_simple_yaml(f.read())
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return config


def save_checkpoint(state: dict[str, Any], path: str | os.PathLike[str]) -> None:
    import torch

    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    torch.save(state, checkpoint_path)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError("PyYAML is not installed and fallback parser only supports simple key: value lines.")
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"null", "none"}:
            parsed_value: Any = None
        elif value.lower() in {"true", "false"}:
            parsed_value = value.lower() == "true"
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value.strip("\"'")
        config[key.strip()] = parsed_value
    return config
