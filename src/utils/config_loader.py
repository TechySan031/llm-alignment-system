"""
Configuration loading utilities.
"""

from __future__ import annotations

from typing import Iterable

# pyrefly: ignore [missing-import]
from omegaconf import OmegaConf


def load_config(path: str):
    """
    Load YAML configuration file as OmegaConf object.
    """

    return OmegaConf.load(path)


def apply_overrides(cfg, overrides: Iterable[str]):
    """
    Apply CLI overrides.

    Example:
        overrides = [
            "training.lr=1e-4",
            "training.batch_size=8",
        ]
    """

    if not overrides:
        return cfg

    override_cfg = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(cfg, override_cfg)


def save_config(cfg, path: str) -> None:
    """
    Save OmegaConf configuration.
    """

    OmegaConf.save(cfg, path)


def config_to_dict(cfg) -> dict:
    """
    Convert OmegaConf to standard Python dict.
    """

    return OmegaConf.to_container(
        cfg,
        resolve=True,
    )