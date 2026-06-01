"""
SFT training pipeline using TRL's SFTTrainer.

Architecture:
    SFTTrainer (TRL)
        ├── Wraps HuggingFace Trainer
        ├── Adds SFT-specific features (chat template, packing)
        ├── Accepts our custom SFTDataCollator
        └── Uses our callbacks for monitoring

Training workflow per step:
    1. DataLoader yields a batch of tokenised examples
    2. SFTDataCollator pads to batch maximum length (dynamic padding)
    3. model.forward(input_ids, attention_mask, labels)
       → logits over vocabulary
       → cross_entropy(logits[..., :-1, :], labels[..., 1:])
         labels=-100 positions are ignored (prompt masking)
    4. loss.backward() → gradients on LoRA A, B matrices only
    5. clip_grad_norm_(trainable_params, max_norm=1.0)
    6. optimizer.step() → AdamW parameter update
    7. scheduler.step() → cosine LR update
    8. optimizer.zero_grad()

Memory breakdown during training on A100 40GB (7B + QLoRA r=16):
    Model weights (NF4):    ~3.8 GB
    LoRA gradients (fp32):  ~0.16 GB
    Optimizer states (paged): 0 GB (on CPU RAM)
    Activations (gc=True):  ~2-4 GB (varies by batch+seq)
    Forward/backward overhead: ~2 GB
    Total:                  ~8-10 GB → fits on 16GB T4 with batch=1

compute_metrics function:
    Called at each eval step with decoded predictions and labels.
    Returns a dict that is logged to W&B as eval metrics.
    We compute json_validity here so you can watch it rise on W&B
    during training — a real-time signal that the model is learning
    the output format.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
# pyrefly: ignore [missing-import]
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, TrainingArguments
# pyrefly: ignore [missing-import]
from trl import SFTConfig, SFTTrainer

# pyrefly: ignore [missing-import]
from src.data.collator import SFTDataCollator
# pyrefly: ignore [missing-import]
from src.training.callbacks import (
    GradientMonitorCallback,
    LearningRateMonitorCallback,
    OverfitDetectorCallback,
    TrainingProgressCallback,
    WandbArtifactCallback,
)
# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_sft_config(cfg: DictConfig, num_training_steps: int = 0) -> SFTConfig:
    """
    Build TRL SFTConfig from Hydra config.

    SFTConfig extends TrainingArguments with:
        max_seq_length:    Hard truncation limit for SFT sequences.
        packing:           Pack multiple short examples into one sequence.
                           Improves GPU utilisation for variable-length data.
                           Disabled here because we pre-tokenise with loss masks
                           and packing would require re-computing masks.
        dataset_kwargs:    skip_prepare_dataset=True because we handle
                           tokenisation ourselves in build_sft_datasets().

    Key training arg decisions:

    gradient_checkpointing=True:
        Recomputes intermediate activations during backward pass instead
        of storing them. Saves ~60% activation VRAM at cost of ~30% compute.
        Essential for fitting 7B models on consumer GPUs.

    use_reentrant=False:
        The non-reentrant gradient checkpointing implementation avoids
        a class of subtle bugs when combined with PEFT. Always use False.

    optim="paged_adamw_32bit":
        Stores optimizer states in CPU RAM. Frees ~320MB GPU VRAM
        for activations or larger batch sizes. ~5% slower per step.

    group_by_length=True:
        Sorts examples by sequence length before batching.
        Sequences of similar length → less padding per batch.
        20-40% throughput improvement on variable-length data.

    bf16 vs fp16:
        bf16=True on Ampere+ (CC >= 8.0) — no loss scaling needed.
        fp16=True on older GPUs (V100, T4).
        Automatically handled by get_training_dtype() in utils.py.
    """
    # pyrefly: ignore [missing-import]
    from src.training.utils import is_bf16_supported

    use_bf16 = cfg.training.get("bf16", is_bf16_supported())
    use_fp16 = cfg.training.get("fp16", False) and not use_bf16

    return SFTConfig(
        # Paths
        output_dir=cfg.training.output_dir,

        # Epochs and steps
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.get("max_steps", -1),

        # Batch sizes
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,

        # Optimisation
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        max_grad_norm=cfg.training.max_grad_norm,
        optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",

        # Precision
        bf16=use_bf16,
        fp16=use_fp16,

        # Memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging and evaluation
        logging_steps=cfg.training.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        eval_on_start=True,  # Baseline eval before any training

        # Checkpointing
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Data loading
        dataloader_num_workers=cfg.training.get("dataloader_num_workers", 0),
        dataloader_pin_memory=torch.cuda.is_available(),
        group_by_length=cfg.training.get("group_by_length", True),
        remove_unused_columns=False,

        # Reporting
        report_to=cfg.training.get("report_to", "none"),
        run_name=cfg.training.get("run_name", "sft-training"),
        logging_first_step=True,

        # Reproducibility
        seed=cfg.training.get("seed", 42),
        data_seed=cfg.training.get("seed", 42),

        # SFT-specific
        max_length=cfg.tokenizer.get("max_length", 2048),
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

def build_compute_metrics(tokenizer: PreTrainedTokenizer):
    """
    Build a compute_metrics function for the SFTTrainer.

    Called after each eval step. Receives EvalPrediction namedtuple:
        predictions: np.ndarray of shape (n, seq, vocab) or (n, seq)
        label_ids:   np.ndarray of shape (n, seq) with -100 for masked positions.

    We decode predictions and labels then compute:
        json_validity: fraction of valid JSON predictions
        This metric rises during training as the model learns the format.

    Note on predictions shape:
        When predict_with_generate=False (default), predictions are logit
        arrays. We take argmax to get token IDs before decoding.
        When predict_with_generate=True, predictions are already token IDs.
    """

    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions, labels = eval_pred

        # Handle both logit arrays and token ID arrays
        if predictions.ndim == 3:
            # Logits: (batch, seq, vocab) → argmax → token IDs
            pred_ids = np.argmax(predictions, axis=-1)
        else:
            pred_ids = predictions

        # Replace -100 in labels with pad token for decoding
        pad_id = tokenizer.pad_token_id or 0
        labels = np.where(labels != -100, labels, pad_id)

        decoded_preds = tokenizer.batch_decode(
            pred_ids.tolist(), skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels.tolist(), skip_special_tokens=True
        )

        # JSON validity rate
        valid_json = 0
        for pred in decoded_preds:
            if _is_valid_json(pred):
                valid_json += 1

        n = max(len(decoded_preds), 1)
        return {
            "json_validity": round(valid_json / n, 4),
        }

    return compute_metrics


def _is_valid_json(text: str) -> bool:
    """Quick JSON validity check with markdown fence recovery."""
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting from markdown fence
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            json.loads(m.group(1).strip())
            return True
        except Exception:
            pass
    # Try first { } span
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            json.loads(m2.group())
            return True
        except Exception:
            pass
    return False


def build_sft_trainer(
    model,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: DictConfig,
    extra_callbacks: Optional[list] = None,
) -> SFTTrainer:
    """
    Assemble the SFTTrainer with all components.

    Args:
        model:           PeftModel with LoRA adapters injected.
        tokenizer:       Matching tokenizer.
        train_dataset:   Tokenised HF Dataset (from build_sft_datasets).
        eval_dataset:    Tokenised HF Dataset (from build_sft_datasets).
        cfg:             Hydra DictConfig with training, lora, tokenizer sections.
        extra_callbacks: Additional callbacks to attach.

    Returns:
        Configured SFTTrainer ready for .train().
    """
    sft_config = build_sft_config(cfg)
    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else 1,
    )

    # Standard callbacks always attached
    callbacks = [
        GradientMonitorCallback(
            log_freq=cfg.training.get("logging_steps", 10)
        ),
        OverfitDetectorCallback(
            divergence_threshold=cfg.training.get("overfit_threshold", 0.35),
            patience=cfg.training.get("overfit_patience", 3),
        ),
        LearningRateMonitorCallback(),
        TrainingProgressCallback(
            log_freq=cfg.training.get("logging_steps", 10) * 2
        ),
    ]

    # W&B artifact callback (only if W&B is configured)
    report_to = cfg.training.get("report_to", "none")
    if report_to == "wandb":
        adapter_path = str(
            Path(cfg.training.output_dir) / "final_adapter"
        )
        callbacks.append(
            WandbArtifactCallback(
                adapter_path=adapter_path,
                run_name=cfg.training.get("run_name", "sft"),
            )
        )

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    # Log effective batch size
    eff_batch = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    logger.info(
        f"[SFTTrainer] Assembled:\n"
        f"  Train examples:         {len(train_dataset):,}\n"
        f"  Eval examples:          {len(eval_dataset):,}\n"
        f"  Per-device batch:       {cfg.training.per_device_train_batch_size}\n"
        f"  Grad accumulation:      {cfg.training.gradient_accumulation_steps}\n"
        f"  Effective batch:        {eff_batch}\n"
        f"  Learning rate:          {cfg.training.learning_rate:.2e}\n"
        f"  Epochs:                 {cfg.training.num_train_epochs}\n"
        f"  Callbacks:              {[type(c).__name__ for c in callbacks]}"
    )
    return trainer