"""
Learning rate scheduler factory.

Cosine with warmup is the standard choice for LLM fine-tuning.

Why warmup is essential:
    At step 0, Adam's moment estimates (m and v) are both zero.
    If you apply the full learning rate immediately, the first few
    updates are very noisy — m/v estimates are unreliable.
    Linear warmup over the first 3% of steps allows the moments
    to accumulate a reliable history before the full LR is applied.
    Without warmup, early training steps can destabilise the model
    and cause loss spikes that are hard to recover from.

Why cosine decay:
    Step decay (halve LR every N epochs) causes abrupt loss spikes
    at each drop. Cosine provides smooth, continuous decay:
        lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t/T))
    The model sees a high LR early (exploration) and a low LR late
    (refinement), without any discontinuities.

Warmup ratio 0.03 means:
    For 1,000 total steps: warmup for 30 steps
    For 5,000 total steps: warmup for 150 steps
    The absolute count scales with total training steps so the
    warmup phase is always 3% of training regardless of dataset size.
"""
from __future__ import annotations

import logging
from typing import Optional

from torch.optim import Optimizer
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    num_cycles: int = 1,
) -> object:
    """
    Build a learning rate scheduler.

    Args:
        optimizer:          The optimizer to attach the scheduler to.
        scheduler_type:     One of: "cosine", "linear", "constant",
                            "cosine_with_restarts".
        num_training_steps: Total number of gradient update steps.
        warmup_ratio:       Fraction of steps used for linear warmup.
        num_cycles:         Number of cosine cycles (only for cosine_with_restarts).

    Returns:
        LambdaLR scheduler.
    """
    num_warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    logger.info(
        f"[Scheduler] type={scheduler_type} | "
        f"total_steps={num_training_steps} | "
        f"warmup={num_warmup_steps} ({warmup_ratio*100:.1f}%)"
    )

    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    elif scheduler_type == "cosine_with_restarts":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
    else:
        logger.warning(
            f"[Scheduler] Unknown type '{scheduler_type}'. "
            "Defaulting to cosine."
        )
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )


def compute_total_steps(
    dataset_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: float,
    num_devices: int = 1,
) -> int:
    """
    Compute total gradient update steps for a training run.

    Formula:
        steps_per_epoch = dataset_size / (per_device_batch × grad_accum × devices)
        total_steps = steps_per_epoch × num_epochs

    One "step" is one optimizer.step() call, which happens after
    accumulating gradients over `gradient_accumulation_steps` micro-batches.
    """
    effective_batch = per_device_batch_size * gradient_accumulation_steps * num_devices
    steps_per_epoch = max(1, dataset_size // effective_batch)
    total = int(steps_per_epoch * num_epochs)

    logger.info(
        f"[Scheduler] Steps: "
        f"dataset={dataset_size} / "
        f"effective_batch={effective_batch} × "
        f"epochs={num_epochs:.1f} = "
        f"{total} total steps "
        f"({steps_per_epoch}/epoch)"
    )
    return total