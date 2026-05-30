"""
Reproducibility and environment utilities.

Provides:
- Random seed setup
- Device detection
- System information logging
"""

from __future__ import annotations

import logging
import os
import platform
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Random seed set to %s", seed)


def get_device() -> torch.device:
    """
    Return best available torch device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_device_name() -> str:
    """
    Human-readable device name.
    """

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return "CPU"


def log_system_info() -> None:
    """
    Log environment information.
    """

    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)

    logger.info("Platform: %s", platform.platform())
    logger.info("Python: %s", platform.python_version())

    logger.info("PyTorch: %s", torch.__version__)

    logger.info(
        "CUDA Available: %s",
        torch.cuda.is_available(),
    )

    logger.info(
        "Device: %s",
        get_device_name(),
    )

    if torch.cuda.is_available():
        logger.info(
            "CUDA Version: %s",
            torch.version.cuda,
        )

        logger.info(
            "GPU Count: %s",
            torch.cuda.device_count(),
        )

    logger.info("=" * 60)


def get_system_info() -> dict:
    """
    Return environment information as dictionary.
    """

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": get_device_name(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()

    return info