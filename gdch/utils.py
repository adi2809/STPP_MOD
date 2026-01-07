import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class SolverConfig:
    method: str = "dopri5"
    rtol: float = 1e-4
    atol: float = 1e-6
    max_num_steps: int = 1000
    use_adjoint: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_dir(root: str, run_name: str) -> str:
    run_dir = os.path.join(root, f"{run_name}_{timestamp()}")
    return ensure_dir(run_dir)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def setup_logger(log_path: Optional[str] = None, name: str = "gdch") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def detach_state(state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if state is None:
        return None
    z, a = state
    return z.detach(), a.detach()


def _parse_override_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(config: Dict[str, Any], overrides: Optional[list]) -> Dict[str, Any]:
    if not overrides:
        return config
    for override in overrides:
        if "=" not in override:
            continue
        key, raw_value = override.split("=", 1)
        value = _parse_override_value(raw_value)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return config
