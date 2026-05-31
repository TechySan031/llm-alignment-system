"""
Training utility functions used across SFT and DPO trainers.
Covers mixed precision setup, checkpoint inspection, and
reproducible training initialisation.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_training_dtype() -> torch.dtype:
    """
    Return the optimal training dtype for the current hardware.

    BF16 on Ampere+ (A100, RTX 30/40xx, H100):
        8 exponent bits = same dynamic range as FP32.
        No loss scaling needed. Fastest on modern GPUs.
        Recommended for all Ampere+ hardware.

    FP16 on Volta/Turing (V100, T4, RTX 20xx):
        5 exponent bits = limited dynamic range.
        Requires gradient scaling to prevent overflow.
        HuggingFace Trainer handles this automatically.

    FP32 on CPU:
        No performance benefit from reduced precision.
        Always use FP32 for CPU training/testing.
    """
    if not torch.cuda.is_available():
        logger.info("[Utils] CPU: using float32")
        return torch.float32

    major = torch.cuda.get_device_properties(0).major
    if major >= 8:
        logger.info(f"[Utils] Ampere+ (CC {major}.x): using bfloat16")
        return torch.bfloat16
    else:
        logger.info(f"[Utils] Pre-Ampere (CC {major}.x): using float16")
        return torch.float16


def is_bf16_supported() -> bool:
    """Return True if the current GPU supports bfloat16."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).major >= 8


def set_training_environment() -> None:
    """
    Configure environment variables for optimal training performance.

    TOKENIZERS_PARALLELISM=false:
        Prevents HuggingFace tokenizers from forking parallelism
        inside PyTorch DataLoader workers (causes deadlocks).

    OMP_NUM_THREADS:
        Limits OpenMP threads per worker. Without this, multiple
        DataLoader workers each spawn many threads, overwhelming
        the CPU and degrading throughput.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # TF32 on Ampere: uses tensor cores for matmul without explicit BF16 cast
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("[Utils] TF32 enabled for matmul and cudnn")


def log_vram_usage(stage: str = "") -> dict:
    """Log and return current VRAM usage. No-op on CPU."""
    if not torch.cuda.is_available():
        return {}
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    stats = {
        "allocated_gb": round(alloc, 3),
        "reserved_gb":  round(reserved, 3),
        "peak_gb":      round(peak, 3),
    }
    label = f"[{stage}] " if stage else ""
    logger.info(
        f"[Utils] {label}VRAM: "
        f"allocated={alloc:.2f}GB | "
        f"reserved={reserved:.2f}GB | "
        f"peak={peak:.2f}GB"
    )
    return stats


def find_resume_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in output_dir for training resumption.

    HuggingFace Trainer saves checkpoints as checkpoint-{step}/.
    Returns the path to the highest-step checkpoint, or None if
    no checkpoints exist.

    Usage:
        resume_from = find_resume_checkpoint("experiments/sft_runs/run_001")
        trainer.train(resume_from_checkpoint=resume_from)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = sorted(
        [
            d for d in output_path.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ],
        key=lambda d: int(d.name.split("-")[1]),
    )

    if not checkpoints:
        return None

    latest = str(checkpoints[-1])
    logger.info(f"[Utils] Found resume checkpoint: {latest}")
    return latest


def count_dataset_tokens(
    dataset,
    sample_size: int = 1000,
) -> dict:
    """
    Estimate token statistics from a tokenised HuggingFace Dataset.

    Args:
        dataset:     Tokenised HF Dataset with 'input_ids' column.
        sample_size: How many examples to sample for statistics.

    Returns:
        Dict with mean, p50, p95, max sequence lengths.
    """
    import numpy as np
    indices = list(range(min(sample_size, len(dataset))))
    lengths = [len(dataset[i]["input_ids"]) for i in indices]
    return {
        "mean":    round(float(np.mean(lengths)), 1),
        "p50":     int(np.percentile(lengths, 50)),
        "p95":     int(np.percentile(lengths, 95)),
        "max":     int(max(lengths)),
        "n_sampled": len(lengths),
    }