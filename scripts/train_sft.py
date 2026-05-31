"""
Phase 5 — Complete SFT training script.

Trains Qwen2.5-7B (or any configured model) with QLoRA using the
dataset from Phase 1 and evaluates against the baseline from Phase 4.

────────────────────────────────────────────────────────────────────────
LOCAL TESTING (Windows AMD — no GPU)
────────────────────────────────────────────────────────────────────────
Uses a small model (0.5B) in float32 on CPU.
Runs 50 steps in ~5 minutes to verify the full pipeline works.

    python scripts/train_sft.py \\
        --config configs/training/sft_lora_local.yaml \\
        --model-config configs/model/qwen2_5_7b.yaml \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset-size 200 \\
        --smoke-test

────────────────────────────────────────────────────────────────────────
GOOGLE COLAB (Free T4 / Pro A100)
────────────────────────────────────────────────────────────────────────
Paste this in a Colab cell:

    !git clone https://github.com/YOUR/llm-alignment-system
    %cd llm-alignment-system
    !pip install -e . -q
    !python scripts/generate_dataset.py --no-public

    # Free T4 (16GB): batch=1, accum=16, max_steps=500
    !python scripts/train_sft.py \\
        --config configs/training/sft_lora.yaml \\
        --model-config configs/model/qwen2_5_7b.yaml \\
        --override training.per_device_train_batch_size=1 \\
        --override training.gradient_accumulation_steps=16 \\
        --override training.max_steps=500 \\
        --wandb-key YOUR_WANDB_KEY

    # Pro A100 (40GB): full training
    !python scripts/train_sft.py \\
        --config configs/training/sft_lora.yaml \\
        --model-config configs/model/qwen2_5_7b.yaml \\
        --wandb-key YOUR_WANDB_KEY

────────────────────────────────────────────────────────────────────────
KAGGLE (2× T4, free)
────────────────────────────────────────────────────────────────────────
In a Kaggle notebook (GPU T4 x2 accelerator):

    import subprocess
    subprocess.run(["pip", "install", "-e", ".", "-q"])
    subprocess.run([
        "python", "scripts/train_sft.py",
        "--config", "configs/training/sft_lora.yaml",
        "--model-config", "configs/model/qwen2_5_7b.yaml",
        "--override", "training.per_device_train_batch_size=1",
        "--override", "training.gradient_accumulation_steps=16",
    ])

────────────────────────────────────────────────────────────────────────
RUNPOD (RTX 4090 24GB ~$0.50/hr)
────────────────────────────────────────────────────────────────────────
Select: PyTorch 2.3 + CUDA 12.1, minimum 24GB VRAM

    git clone https://github.com/YOUR/llm-alignment-system
    cd llm-alignment-system
    pip install -e .
    python scripts/generate_dataset.py --no-public
    python scripts/train_sft.py \\
        --config configs/training/sft_lora.yaml \\
        --model-config configs/model/qwen2_5_7b.yaml \\
        --override training.per_device_train_batch_size=2 \\
        --override training.gradient_accumulation_steps=8

────────────────────────────────────────────────────────────────────────
VAST.AI (A100 80GB, cheapest rates)
────────────────────────────────────────────────────────────────────────
Search: A100 80GB, PyTorch template. Same commands as RunPod.
At batch=4, accum=4 on A100: ~2.5 hours for full 3-epoch training.

────────────────────────────────────────────────────────────────────────
EXPECTED OUTPUTS
────────────────────────────────────────────────────────────────────────
    experiments/sft_runs/run_001/
        checkpoint-200/              saved every save_steps
        checkpoint-400/
        final_adapter/               adapter_model.bin + adapter_config.json
        trainer_state.json           loss history for plotting
        training_args.bin

    metrics/sft_metrics.json         evaluation results
    experiments/benchmark_results/
        benchmark_sft.json           benchmark for comparison
    metrics/benchmark_history.csv    updated with SFT row
    outputs/logs/training/train_sft_*.log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
# pyrefly: ignore [missing-import]
from omegaconf import DictConfig, OmegaConf

# pyrefly: ignore [missing-import]
from src.data.generator import DatasetOrchestrator
# pyrefly: ignore [missing-import]
from src.data.preprocessor import build_sft_datasets
# pyrefly: ignore [missing-import]
from src.data.schemas import ModelStage, SFTExample
# pyrefly: ignore [missing-import]
from src.evaluation.benchmarks import EvaluationPipeline
# pyrefly: ignore [missing-import]
from src.evaluation.comparator import ModelComparator
# pyrefly: ignore [missing-import]
from src.models.lora_config import prepare_model_for_training
# pyrefly: ignore [missing-import]
from src.models.loader import load_base_model
# pyrefly: ignore [missing-import]
from src.models.parameter_counter import lora_efficiency_report, print_parameter_table
# pyrefly: ignore [missing-import]
from src.training.sft_trainer import build_sft_trainer
# pyrefly: ignore [missing-import]
from src.training.utils import (
    log_vram_usage,
    set_training_environment,
    find_resume_checkpoint,
    count_dataset_tokens,
)
# pyrefly: ignore [missing-import]
from src.utils.config_loader import load_config, apply_overrides
# pyrefly: ignore [missing-import]
from src.utils.file_utils import (
    ensure_dir,
    read_jsonl_all,
    save_metrics,
    write_json,
)
# pyrefly: ignore [missing-import]
from src.utils.logging import log_dict, log_section, setup_logging
# pyrefly: ignore [missing-import]
from src.utils.reproducibility import log_system_info, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5 — SFT Training with QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/training/sft_lora.yaml",
        help="Training config YAML path",
    )
    p.add_argument(
        "--model-config",
        type=str,
        default="configs/model/qwen2_5_7b.yaml",
        help="Model config YAML path",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from config",
    )
    p.add_argument(
        "--dataset-size",
        type=int,
        default=0,
        help="Override dataset size. 0 = use existing data/processed/",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values. E.g. --override training.learning_rate=1e-4",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output_dir",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick 50-step test run. Overrides num_train_epochs and max_steps.",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip post-training evaluation (faster, use for ablations)",
    )
    p.add_argument(
        "--wandb-key",
        type=str,
        default=None,
        help="W&B API key (alternative to setting in .env)",
    )
    p.add_argument(
        "--load-public",
        action="store_true",
        default=False,
        help="Load public datasets (UltraChat + OpenHermes). Requires internet.",
    )
    return p.parse_args()


def setup_wandb(wandb_key: Optional[str], cfg: DictConfig) -> None:
    """Initialise W&B if configured."""
    if cfg.training.get("report_to", "none") == "none":
        return
    try:
        # pyrefly: ignore [missing-import]
        import wandb
        import os
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "llm-alignment-system"),
            name=cfg.training.get("run_name", "sft-training"),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["sft", "qlora", cfg.model.name.split("/")[-1]],
            resume="allow",
        )
        logger = logging.getLogger(__name__)
        logger.info(f"[W&B] Run: {wandb.run.url}")
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("wandb not installed — training without experiment tracking")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"W&B init failed: {e}")


def load_or_generate_datasets(
    cfg: DictConfig,
    tokenizer,
    dataset_size: int,
    load_public: bool,
) -> tuple:
    """
    Load pre-generated datasets from disk, or generate fresh ones.

    If data/processed/train.jsonl exists: load from disk (fast).
    If not, or if dataset_size > 0: generate fresh synthetic data.

    This allows:
        - Quick iteration: generate once, train many times
        - Experiment with different dataset sizes via --dataset-size
        - Always use the same test set for fair comparison
    """
    logger = logging.getLogger(__name__)
    train_path = Path("data/processed/train.jsonl")
    val_path = Path("data/processed/validation.jsonl")

    if dataset_size > 0 or not train_path.exists():
        logger.info(
            f"Generating {'new' if dataset_size > 0 else 'missing'} dataset "
            f"(size={dataset_size or 'default'}, load_public={load_public})"
        )
        orchestrator = DatasetOrchestrator(
            seed=cfg.training.get("seed", 42),
            load_public=load_public,
        )
        sft_examples = orchestrator.generate_sft_examples()
        if dataset_size > 0:
            sft_examples = sft_examples[:dataset_size]
        return build_sft_datasets(
            sft_examples=sft_examples,
            tokenizer=tokenizer,
            max_length=cfg.tokenizer.get("max_length", 2048),
            seed=cfg.training.get("seed", 42),
        )

    # Load from disk
    logger.info(f"Loading existing datasets from data/processed/")

    raw_train = read_jsonl_all(str(train_path))
    raw_val = read_jsonl_all(str(val_path))
    raw_test = (
        read_jsonl_all("data/processed/test.jsonl")
        if Path("data/processed/test.jsonl").exists()
        else []
    )

    train_examples = [SFTExample.from_dict(r) for r in raw_train]
    val_examples = [SFTExample.from_dict(r) for r in raw_val]
    test_examples = [SFTExample.from_dict(r) for r in raw_test]

    logger.info(
        f"Loaded: train={len(train_examples):,}, "
        f"val={len(val_examples):,}, "
        f"test={len(test_examples):,}"
    )

    return build_sft_datasets(
        sft_examples=train_examples + val_examples + test_examples,
        tokenizer=tokenizer,
        max_length=cfg.tokenizer.get("max_length", 2048),
        seed=cfg.training.get("seed", 42),
    )


def main() -> None:
    args = parse_args()

    logger = setup_logging(
        level="INFO",
        log_dir="outputs/logs/training",
        run_name=f"train_sft",
    )

    set_training_environment()

    # ── Load and merge configs ────────────────────────────────────────────────
    log_section(logger, "Loading configuration")
    cfg = load_config(args.config)
    model_cfg = load_config(args.model_config)
    cfg = OmegaConf.merge(model_cfg, cfg)

    # Apply CLI overrides
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    # Override model if specified
    if args.model:
        cfg.model.name = args.model
        logger.info(f"Model override: {args.model}")

    # Smoke test overrides
    if args.smoke_test:
        cfg.training.max_steps = 50
        cfg.training.num_train_epochs = 1
        cfg.training.eval_steps = 25
        cfg.training.save_steps = 50
        cfg.training.logging_steps = 5
        cfg.training.report_to = "none"
        cfg.tokenizer.max_length = 512
        logger.info("[SmokeTest] max_steps=50, max_length=512")

    log_dict(
        logger,
        {
            "model":          cfg.model.name,
            "lora_r":         cfg.lora.r,
            "lora_alpha":     cfg.lora.lora_alpha,
            "lr":             cfg.training.learning_rate,
            "batch":          cfg.training.per_device_train_batch_size,
            "grad_accum":     cfg.training.gradient_accumulation_steps,
            "effective_batch": (
                cfg.training.per_device_train_batch_size
                * cfg.training.gradient_accumulation_steps
            ),
            "epochs":         cfg.training.num_train_epochs,
            "max_steps":      cfg.training.get("max_steps", -1),
            "max_length":     cfg.tokenizer.max_length,
            "output_dir":     cfg.training.output_dir,
            "report_to":      cfg.training.get("report_to", "none"),
        },
        "Training configuration",
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = cfg.training.get("seed", 42)
    set_seed(seed)
    system_info = log_system_info()

    # ── W&B ───────────────────────────────────────────────────────────────────
    setup_wandb(args.wandb_key, cfg)

    # ── Create output directories ─────────────────────────────────────────────
    ensure_dir(cfg.training.output_dir)
    ensure_dir("experiments/benchmark_results")
    ensure_dir("metrics")

    # ── Load model ────────────────────────────────────────────────────────────
    log_section(logger, "Loading model")
    log_vram_usage("before_model_load")

    try:
        model, tokenizer = load_base_model(cfg)
    except Exception as e:
        logger.error(
            f"Model load failed: {e}\n\n"
            "For local testing use:\n"
            "  python scripts/train_sft.py \\\n"
            "    --config configs/training/sft_lora_local.yaml \\\n"
            "    --model-config configs/model/qwen2_5_7b.yaml \\\n"
            "    --model Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "    --smoke-test"
        )
        sys.exit(1)

    log_vram_usage("after_model_load")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    log_section(logger, "Applying LoRA adapters")

    is_quantized = cfg.quantization.get("load_in_4bit", False) and torch.cuda.is_available()
    model = prepare_model_for_training(
        model, cfg, is_quantized=is_quantized
    )

    # Verify and log parameter efficiency
    print_parameter_table(model)
    efficiency = lora_efficiency_report(model)
    log_dict(logger, efficiency, "LoRA efficiency report")
    log_vram_usage("after_lora_injection")

    # Log to W&B
    try:
        # pyrefly: ignore [missing-import]
        import wandb
        if wandb.run:
            wandb.config.update({
                "trainable_params_M":  efficiency["trainable_params_M"],
                "efficiency_ratio":    efficiency["efficiency_ratio"],
                "param_savings_pct":   efficiency["param_savings_pct"],
                **{f"system/{k}": v for k, v in system_info.items()
                   if not isinstance(v, dict)},
            })
    except ImportError:
        pass

    # ── Build datasets ────────────────────────────────────────────────────────
    log_section(logger, "Building datasets")
    train_ds, val_ds, test_ds = load_or_generate_datasets(
        cfg=cfg,
        tokenizer=tokenizer,
        dataset_size=args.dataset_size,
        load_public=args.load_public,
    )

    # Log sequence length statistics
    token_stats = count_dataset_tokens(train_ds)
    log_dict(logger, token_stats, "Train sequence length statistics")

    try:
        # pyrefly: ignore [missing-import]
        import wandb
        if wandb.run:
            wandb.config.update({
                "train_size":         len(train_ds),
                "val_size":           len(val_ds),
                "test_size":          len(test_ds),
                "mean_seq_length":    token_stats["mean"],
                "p95_seq_length":     token_stats["p95"],
            })
    except ImportError:
        pass

    # ── Build trainer ─────────────────────────────────────────────────────────
    log_section(logger, "Assembling SFTTrainer")
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        cfg=cfg,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    log_section(logger, "Starting SFT training")

    resume_from = None
    if args.resume:
        resume_from = find_resume_checkpoint(cfg.training.output_dir)
        if resume_from:
            logger.info(f"Resuming from: {resume_from}")
        else:
            logger.info("No checkpoint found — starting from scratch")

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current checkpoint...")
        trainer.save_model(str(Path(cfg.training.output_dir) / "interrupted_adapter"))
        sys.exit(0)
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "CUDA OOM during training.\n"
            "Solutions:\n"
            "  1. Reduce per_device_train_batch_size: --override training.per_device_train_batch_size=1\n"
            "  2. Reduce max_length: --override tokenizer.max_length=1024\n"
            "  3. Reduce LoRA rank: --override lora.r=8\n"
            "  4. Use a smaller model: --model Qwen/Qwen2.5-0.5B-Instruct"
        )
        sys.exit(1)

    # ── Save final adapter ────────────────────────────────────────────────────
    log_section(logger, "Saving LoRA adapter")
    adapter_path = Path(cfg.training.output_dir) / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # Save training config alongside adapter for reproducibility
    write_json(
        OmegaConf.to_container(cfg, resolve=True),
        str(adapter_path / "training_config.json"),
    )

    # pyrefly: ignore [missing-import]
    from src.utils.file_utils import get_dir_size_mb
    adapter_size = get_dir_size_mb(str(adapter_path))
    logger.info(
        f"Adapter saved → {adapter_path} "
        f"({adapter_size:.1f} MB)"
    )

    # ── Training summary ──────────────────────────────────────────────────────
    metrics = train_result.metrics
    train_summary = {
        "train_loss":       round(metrics.get("train_loss", 0), 4),
        "train_runtime_h":  round(metrics.get("train_runtime", 0) / 3600, 2),
        "train_samples_s":  round(metrics.get("train_samples_per_second", 0), 2),
        "total_steps":      metrics.get("total_flos", 0),
        "adapter_size_mb":  round(adapter_size, 1),
        "adapter_path":     str(adapter_path),
        "best_eval_loss":   round(trainer.state.best_metric or 0, 4),
    }
    log_dict(logger, train_summary, "Training summary")

    try:
        # pyrefly: ignore [missing-import]
        import wandb
        if wandb.run:
            wandb.log({f"final/{k}": v for k, v in train_summary.items()})
    except ImportError:
        pass

    # ── Post-training evaluation ───────────────────────────────────────────────
    if not args.no_eval:
        log_section(logger, "Post-training evaluation")
        logger.info(
            "Loading test examples for SFT evaluation...\n"
            "This uses the same test split as the baseline — "
            "fair comparison guaranteed."
        )

        test_examples: list[SFTExample] = []
        if Path("data/processed/test.jsonl").exists():
            for record in read_jsonl_all("data/processed/test.jsonl"):
                try:
                    test_examples.append(SFTExample.from_dict(record))
                except Exception:
                    continue

        if test_examples:
            # Use best checkpoint for evaluation
            model.eval()

            eval_pipeline = EvaluationPipeline(
                model=model,
                tokenizer=tokenizer,
                model_stage=ModelStage.SFT,
                model_id=str(adapter_path),
            )

            sft_result = eval_pipeline.run(
                test_examples=test_examples,
                output_dir="experiments/benchmark_results",
                max_examples=200 if args.smoke_test else None,
                save_predictions=True,
            )

            # Save SFT metrics
            save_metrics(
                metrics=sft_result.to_dict(),
                path="metrics/sft_metrics.json",
                run_name=cfg.training.get("run_name", "sft"),
            )

            # Compare with baseline
            comparator = ModelComparator()
            comparator.add(sft_result)

            baseline_path = "experiments/benchmark_results/benchmark_base.json"
            if Path(baseline_path).exists():
                comparator.add_from_file(baseline_path)
                log_section(logger, "SFT vs Baseline Comparison")
                comparator.print_comparison_table()
            else:
                logger.info(
                    "No baseline found. Run scripts/run_baseline.py "
                    "to generate the baseline first."
                )

            comparator.update_history("metrics/benchmark_history.csv")

            try:
                # pyrefly: ignore [missing-import]
                import wandb
                if wandb.run:
                    wandb.log({
                        "sft_eval/format_valid":       sft_result.format_valid,
                        "sft_eval/schema_compliant":   sft_result.schema_compliant,
                        "sft_eval/instruction_followed": sft_result.instruction_followed,
                        "sft_eval/hallucination_rate": sft_result.hallucination_rate,
                        "sft_eval/avg_field_f1":       sft_result.avg_field_f1,
                        "sft_eval/avg_alignment_score": sft_result.avg_alignment_score,
                        "sft_eval/avg_latency_ms":     sft_result.avg_latency_ms,
                    })
            except ImportError:
                pass
        else:
            logger.warning("No test examples found — skipping evaluation")

    # ── Finish ────────────────────────────────────────────────────────────────
    try:
        # pyrefly: ignore [missing-import]
        import wandb
        if wandb.run:
            wandb.finish()
    except ImportError:
        pass

    log_section(logger, "Phase 5 Complete")
    logger.info(
        f"Adapter:  {adapter_path}\n"
        f"Metrics:  metrics/sft_metrics.json\n"
        f"History:  metrics/benchmark_history.csv\n\n"
        f"Next step — Phase 6 (DPO alignment):\n"
        f"  python scripts/train_dpo.py \\\n"
        f"    --sft-adapter {adapter_path}"
    )


if __name__ == "__main__":
    main()