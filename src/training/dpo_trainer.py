"""
DPO training pipeline using TRL's DPOTrainer.

Architecture:
    DPOTrainer (TRL)
        ├── Policy model:    SFT model + LoRA adapters (trainable)
        ├── Reference model: Frozen SFT model (no gradients)
        ├── Dataset:         (prompt, chosen, rejected) triples
        └── Loss:            DPO sigmoid loss

Reference model options:

Option A — Explicit frozen copy (memory-expensive):
    Load the SFT model twice: once as policy (trainable) and once
    as reference (frozen). Requires 2× model VRAM.
    Used when you want the absolute cleanest reference logits.

Option B — Implicit reference via PEFT (memory-efficient, default):
    TRL supports using a PEFT model where the base weights act as the
    reference and the adapter weights are the policy delta.
    Pass ref_model=None and DPOConfig uses the base model layers
    (no adapter) as the implicit reference.
    Saves ~4GB VRAM on a 7B QLoRA model.

Option C — Reference model on CPU (recommended for 16GB GPUs):
    Load reference model in float32 on CPU. Reference forward passes
    are slower but free GPU VRAM entirely for policy training.
    Set ref_model_device="cpu" in DPOConfig.

We use Option B by default (ref_model=None) and fall back to
Option A when the caller explicitly passes a reference model.

DPO β values and their effect:
    β = 0.01: Almost no KL constraint. Policy drifts aggressively
              from SFT. Risk of reward hacking and collapsed outputs.
    β = 0.1:  Standard. Moderate alignment signal. Good default.
    β = 0.5:  Stronger KL. Policy stays closer to SFT. Less alignment.
    β = 1.0:  Very conservative. Almost no change from SFT. Mostly
              useful for weak alignment when SFT is already good.

Loss types:
    sigmoid (standard DPO): -log σ(β · (r_w - r_l))
    ipo:    (r_w - r_l - 1/β)²   — more stable, less sensitive to β
    kto_pair: KTO formulation with separate chosen/rejected losses
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict
# pyrefly: ignore [missing-import]
from omegaconf import DictConfig
# pyrefly: ignore [missing-import]
from trl import DPOConfig, DPOTrainer

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


def build_dpo_config(cfg: DictConfig) -> DPOConfig:
    """
    Build TRL DPOConfig from Hydra config.

    DPOConfig extends TrainingArguments with DPO-specific parameters.
    Most training parameters mirror SFT (batch size, LR, etc.) but
    DPO typically needs a lower LR since the model has already been
    fine-tuned by SFT — we are applying a second, gentler adjustment.

    Typical DPO hyperparameters vs SFT:
        SFT lr:  2e-4  →  DPO lr:  5e-5  (4× smaller)
        SFT epochs: 3  →  DPO epochs: 1  (less data, risk overfitting)
        SFT warmup: 3% →  DPO warmup: 5% (longer warmup for stability)
    """
    # pyrefly: ignore [missing-import]
    from src.training.utils import is_bf16_supported

    use_bf16 = cfg.training.get("bf16", is_bf16_supported())
    use_fp16 = cfg.training.get("fp16", False) and not use_bf16

    return DPOConfig(
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
        optim="adamw_torch",

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

        # Checkpointing
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Data loading
        dataloader_num_workers=cfg.training.get("dataloader_num_workers", 0),
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,

        # Reporting
        report_to=cfg.training.get("report_to", "none"),
        run_name=cfg.training.get("run_name", "dpo-training"),
        logging_first_step=True,

        # Reproducibility
        seed=cfg.training.get("seed", 42),
        data_seed=cfg.training.get("seed", 42),

        # DPO-specific
        max_length=cfg.dpo.max_length,
        beta=cfg.dpo.beta,
        label_smoothing=cfg.dpo.label_smoothing,
        loss_type=cfg.dpo.loss_type,
    )


def build_dpo_datasets_from_disk(
    dpo_train_path: str,
    dpo_val_path: str,
) -> DatasetDict:
    """
    Load DPO preference pairs from JSONL files into a DatasetDict.

    Expected JSONL format (one per line):
        {"prompt": "...", "chosen": "...", "rejected": "..."}

    This is exactly what DPOTrainer expects. The prompt must already
    include the full chat template prefix up to the assistant turn start
    (add_generation_prompt=True was applied during data generation).

    Falls back gracefully if val file does not exist.
    """
    # pyrefly: ignore [missing-import]
    from src.utils.file_utils import read_jsonl_all

    train_records = read_jsonl_all(dpo_train_path)
    train_pairs = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in train_records
        if all(k in r for k in ["prompt", "chosen", "rejected"])
    ]

    val_pairs = []
    if Path(dpo_val_path).exists():
        val_records = read_jsonl_all(dpo_val_path)
        val_pairs = [
            {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in val_records
            if all(k in r for k in ["prompt", "chosen", "rejected"])
        ]

    logger.info(
        f"[DPO Data] Loaded: "
        f"train={len(train_pairs):,}, val={len(val_pairs):,}"
    )

    datasets: dict = {"train": Dataset.from_list(train_pairs)}
    if val_pairs:
        datasets["validation"] = Dataset.from_list(val_pairs)
    else:
        # Use 10% of train as val if no val file
        split = int(len(train_pairs) * 0.9)
        datasets["train"] = Dataset.from_list(train_pairs[:split])
        datasets["validation"] = Dataset.from_list(train_pairs[split:])
        logger.info(
            "[DPO Data] No val file — split train 90/10: "
            f"train={split}, val={len(train_pairs)-split}"
        )

    return DatasetDict(datasets)


def build_dpo_datasets_from_sft_examples(
    sft_train_examples,
    tokenizer,
    seed: int = 42,
) -> DatasetDict:
    """
    Build DPO pairs on-the-fly from SFTExample objects.

    Used when DPO JSONL files have not been pre-generated.
    Calls build_dpo_datasets() from preprocessor.py.
    """
    # pyrefly: ignore [missing-import]
    from src.data.preprocessor import build_dpo_datasets
    return build_dpo_datasets(sft_train_examples, tokenizer, seed=seed)


def build_dpo_trainer(
    model,
    ref_model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: DictConfig,
    extra_callbacks: Optional[list] = None,
) -> DPOTrainer:
    """
    Assemble the DPOTrainer with callbacks and config.

    Args:
        model:         SFT model with LoRA adapters (trainable policy).
                       If LoRA is applied, the base weights serve as
                       the implicit reference when ref_model=None.
        ref_model:     Explicit reference model (frozen SFT model).
                       Pass None to use implicit reference via PEFT.
                       None is strongly recommended to save VRAM.
        tokenizer:     Matching tokenizer.
        train_dataset: Dataset with prompt/chosen/rejected columns.
        eval_dataset:  Dataset with prompt/chosen/rejected columns.
        cfg:           Hydra DictConfig with training and dpo sections.
        extra_callbacks: Additional training callbacks.

    Returns:
        Configured DPOTrainer ready for .train().
    """
    dpo_config = build_dpo_config(cfg)

    callbacks = [
        GradientMonitorCallback(
            log_freq=cfg.training.get("logging_steps", 10)
        ),
        OverfitDetectorCallback(
            divergence_threshold=cfg.training.get("overfit_threshold", 0.4),
            patience=cfg.training.get("overfit_patience", 3),
        ),
        LearningRateMonitorCallback(),
        TrainingProgressCallback(
            log_freq=cfg.training.get("logging_steps", 10) * 2
        ),
    ]

    report_to = cfg.training.get("report_to", "none")
    if report_to == "wandb":
        adapter_path = str(Path(cfg.training.output_dir) / "final_adapter")
        callbacks.append(
            WandbArtifactCallback(
                adapter_path=adapter_path,
                run_name=cfg.training.get("run_name", "dpo"),
            )
        )

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    eff_batch = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
    )
    logger.info(
        f"[DPOTrainer] Assembled:\n"
        f"  Train pairs:       {len(train_dataset):,}\n"
        f"  Eval pairs:        {len(eval_dataset):,}\n"
        f"  Reference model:   {'explicit' if ref_model else 'implicit (PEFT base)'}\n"
        f"  β (beta):          {cfg.dpo.beta}\n"
        f"  Loss type:         {cfg.dpo.get('loss_type', 'sigmoid')}\n"
        f"  Max prompt length: {cfg.dpo.get('max_prompt_length', 1024)}\n"
        f"  Max total length:  {cfg.dpo.get('max_length', 2048)}\n"
        f"  Effective batch:   {eff_batch}\n"
        f"  Learning rate:     {cfg.training.learning_rate:.2e}"
    )
    return trainer