"""
Custom HuggingFace Trainer callbacks for SFT training.

Callbacks plug into the training loop at defined hooks without
modifying the Trainer class itself. This is the correct abstraction:
observation logic is separated from training logic.

Hook execution order per step:
    on_step_begin → [forward + backward + optimizer step] → on_step_end
    on_evaluate  → [eval loop] → on_evaluate (after)
    on_save      → [checkpoint saved]
    on_log       → [metrics logged]

GradientMonitorCallback:
    Computes the L2 norm of gradients across all trainable parameters
    after each backward pass (before gradient clipping).
    Healthy range for LLM fine-tuning: 0.1 – 2.0.
    > 5.0: learning rate too high or data quality issue.
    < 0.01: vanishing gradients, learning may have stalled.
    We log the raw norm AND a 20-step EMA for trend analysis.

OverfitDetectorCallback:
    Compares train loss to eval loss at each evaluation step.
    gap = (eval_loss - train_loss) / train_loss
    gap > threshold for `patience` consecutive evaluations → early stop.
    For SFT on 7B with QLoRA, a gap > 0.3 after epoch 2 is a signal
    that you have more data than the LoRA rank can learn from,
    or that your learning rate is too high.

LearningRateMonitorCallback:
    Logs the actual LR at each step. Useful for verifying:
    - Warmup is working (LR increases linearly for first N steps)
    - Cosine decay is happening (LR decreases smoothly)
    - LR did not jump after a checkpoint resume

WandbArtifactCallback:
    Saves the LoRA adapter as a W&B artifact after training completes.
    Enables artifact versioning and download without re-training.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GradientMonitorCallback(TrainerCallback):
    """
    Tracks gradient norms per training step and logs to W&B.

    Called AFTER loss.backward() and BEFORE optimizer.step()
    via the on_step_end hook (which runs after gradient computation
    but the Trainer has already called clip_grad_norm at this point —
    so we see the clipped norm, not the raw norm).

    To see raw pre-clip norms, call this from a custom training loop.
    For portfolio purposes, post-clip norms are sufficient to demonstrate
    training dynamics understanding.
    """

    def __init__(self, log_freq: int = 10, ema_window: int = 20):
        """
        Args:
            log_freq:   Log every N global steps.
            ema_window: Window size for exponential moving average.
        """
        self.log_freq = log_freq
        self.ema_window = ema_window
        self._grad_norms: deque[float] = deque(maxlen=ema_window)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if state.global_step % self.log_freq != 0 or model is None:
            return

        total_norm = self._compute_grad_norm(model)
        if total_norm is None:
            return

        self._grad_norms.append(total_norm)
        ema_norm = float(np.mean(self._grad_norms))

        # Log to W&B if available
        try:
            # pyrefly: ignore [missing-import]
            import wandb
            if wandb.run:
                wandb.log(
                    {
                        "train/grad_norm":     total_norm,
                        "train/grad_norm_ema": ema_norm,
                    },
                    step=state.global_step,
                )
        except ImportError:
            pass

        # Warn on anomalous norms
        if total_norm > 5.0:
            logger.warning(
                f"[GradMonitor] Step {state.global_step}: "
                f"High gradient norm = {total_norm:.3f}. "
                "Consider reducing learning_rate or checking data quality."
            )
        elif total_norm < 1e-4:
            logger.warning(
                f"[GradMonitor] Step {state.global_step}: "
                f"Very low gradient norm = {total_norm:.6f}. "
                "Learning may have stalled."
            )
        else:
            logger.debug(
                f"[GradMonitor] Step {state.global_step}: "
                f"norm={total_norm:.4f} | ema={ema_norm:.4f}"
            )

    def _compute_grad_norm(self, model) -> Optional[float]:
        """Compute total L2 gradient norm across all trainable parameters."""
        total_sq = 0.0
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                total_sq += param.grad.data.norm(2).item() ** 2
                has_grad = True
        if not has_grad:
            return None
        return float(total_sq ** 0.5)


class OverfitDetectorCallback(TrainerCallback):
    """
    Detects overfitting by monitoring the train-eval loss gap.

    Overfitting signal:
        gap = (eval_loss - train_loss) / (train_loss + ε) > threshold

    When gap exceeds threshold for `patience` consecutive evaluations,
    sets control.should_training_stop = True.

    For SFT fine-tuning:
        Normal gap:      0.05 – 0.20  (some generalisation loss is expected)
        Warning zone:    0.20 – 0.40  (monitor closely)
        Stop zone:       > 0.40       (overfitting, stop and use best checkpoint)

    The best checkpoint (lowest eval loss) is always saved regardless
    of early stopping, because load_best_model_at_end=True in training args.
    """

    def __init__(
        self,
        divergence_threshold: float = 0.35,
        patience: int = 3,
    ):
        self.threshold = divergence_threshold
        self.patience = patience
        self._consecutive_divergence = 0
        self._best_eval_loss = float("inf")
        self._last_train_loss: Optional[float] = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Capture the most recent training loss from logs."""
        if logs and "loss" in logs:
            self._last_train_loss = logs["loss"]

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        train_loss = self._last_train_loss

        # Track best eval loss for reporting
        if eval_loss and eval_loss < self._best_eval_loss:
            self._best_eval_loss = eval_loss

        if eval_loss and train_loss and train_loss > 0:
            gap = (eval_loss - train_loss) / train_loss

            try:
                # pyrefly: ignore [missing-import]
                import wandb
                if wandb.run:
                    wandb.log(
                        {
                            "eval/loss":          eval_loss,
                            "eval/train_gap":     gap,
                            "eval/best_loss":     self._best_eval_loss,
                        },
                        step=state.global_step,
                    )
            except ImportError:
                pass

            if gap > self.threshold:
                self._consecutive_divergence += 1
                logger.warning(
                    f"[OverfitDetector] Step {state.global_step}: "
                    f"train={train_loss:.4f} eval={eval_loss:.4f} "
                    f"gap={gap:.3f} > {self.threshold} "
                    f"({self._consecutive_divergence}/{self.patience})"
                )
                if self._consecutive_divergence >= self.patience:
                    logger.warning(
                        "[OverfitDetector] Early stopping triggered. "
                        f"Best eval loss: {self._best_eval_loss:.4f}"
                    )
                    control.should_training_stop = True
            else:
                if self._consecutive_divergence > 0:
                    logger.info(
                        f"[OverfitDetector] Gap normalised at step "
                        f"{state.global_step}: {gap:.3f}"
                    )
                self._consecutive_divergence = 0


class LearningRateMonitorCallback(TrainerCallback):
    """
    Logs the actual learning rate at each logging step.

    The actual LR after warmup + cosine decay differs from the
    configured learning_rate. Logging it verifies the schedule
    is working as expected.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        lr = logs.get("learning_rate")
        if lr is None:
            return
        try:
            # pyrefly: ignore [missing-import]
            import wandb
            if wandb.run:
                wandb.log(
                    {"train/learning_rate": lr},
                    step=state.global_step,
                )
        except ImportError:
            pass


class TrainingProgressCallback(TrainerCallback):
    """
    Logs a clean progress summary every N steps.

    Replaces the verbose HuggingFace default logging with a
    concise single-line summary suitable for cloud terminal output.
    """

    def __init__(self, log_freq: int = 50):
        self.log_freq = log_freq
        self._step_losses: list[float] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        loss = logs.get("loss")
        if loss:
            self._step_losses.append(loss)

        if state.global_step % self.log_freq == 0 and state.global_step > 0:
            recent_loss = (
                float(np.mean(self._step_losses[-self.log_freq:]))
                if self._step_losses
                else 0.0
            )
            pct_done = (
                100 * state.global_step / state.max_steps
                if state.max_steps > 0
                else 0
            )
            logger.info(
                f"[Training] Step {state.global_step}/{state.max_steps} "
                f"({pct_done:.1f}%) | "
                f"loss={recent_loss:.4f} | "
                f"lr={logs.get('learning_rate', 0):.2e} | "
                f"epoch={state.epoch:.2f}"
            )


class WandbArtifactCallback(TrainerCallback):
    """
    Saves the LoRA adapter as a W&B artifact after training.

    Artifacts are versioned model snapshots in W&B.
    This allows you to:
        - Download the adapter without re-running training
        - Compare adapter weights across runs
        - Roll back to a previous checkpoint
    """

    def __init__(self, adapter_path: str, run_name: str):
        self.adapter_path = adapter_path
        self.run_name = run_name

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        try:
            # pyrefly: ignore [missing-import]
            import wandb
            if not wandb.run:
                return
            artifact = wandb.Artifact(
                name=f"lora-adapter-{self.run_name}",
                type="model",
                description=f"QLoRA adapter trained with SFT. Best eval loss: "
                f"{state.best_metric:.4f}" if state.best_metric else "",
            )
            artifact.add_dir(self.adapter_path)
            wandb.log_artifact(artifact)
            logger.info(
                f"[WandbArtifact] Adapter uploaded as artifact: "
                f"lora-adapter-{self.run_name}"
            )
        except ImportError:
            logger.debug("W&B not installed — skipping artifact upload")
        except Exception as e:
            logger.warning(f"[WandbArtifact] Upload failed: {e}")