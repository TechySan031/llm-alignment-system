"""
Phase 6 — Complete DPO alignment training script.

Loads the SFT adapter from Phase 5, applies DPO preference optimization
using the preference pairs from Phase 1, evaluates the result against
both the base model baseline and the SFT model.

────────────────────────────────────────────────────────────────────────
HOW DPO TRAINING WORKS IN PRACTICE
────────────────────────────────────────────────────────────────────────

1. The policy model is the SFT model with NEW LoRA adapters initialised.
   (We do NOT continue training the SFT adapters — we create fresh ones.)
   Why: The SFT adapters are the reference point. We want to measure
   policy drift from SFT, not from base. Using the SFT adapters as
   the starting point for DPO adapters conflates the two training stages.

   In practice with TRL + PEFT:
   - Load SFT checkpoint as base (merge or load with is_trainable=True)
   - Apply fresh LoRA adapters to it
   - The PEFT base weights become the implicit reference
   - The new LoRA adapters are the policy delta

2. Each DPO training step processes a (prompt, chosen, rejected) triple:
   a. Forward pass through policy model for both chosen and rejected
   b. Forward pass through reference model for both (no grad)
   c. Compute log-ratios: log(π_θ/π_ref) for chosen and rejected
   d. DPO loss = -log σ(β · (log_ratio_chosen - log_ratio_rejected))
   e. Backward through policy only

3. What decreases during training:
   - eval_loss: mean DPO loss across eval pairs
   - rewards/rejected: policy's relative log-prob for rejected responses
   What increases:
   - rewards/chosen: policy's relative log-prob for chosen responses
   - rewards/margins: chosen - rejected (the alignment gap)
   - rewards/accuracies: fraction of pairs where chosen > rejected

────────────────────────────────────────────────────────────────────────
LOCAL TESTING (Windows AMD — CPU)
────────────────────────────────────────────────────────────────────────
    python scripts/train_dpo.py \\
        --config configs/training/dpo_local.yaml \\
        --model-config configs/model/qwen2_5_7b.yaml \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --sft-adapter experiments/sft_runs/local_test/final_adapter \\
        --smoke-test

────────────────────────────────────────────────────────────────────────
GOOGLE COLAB (T4 16GB)
────────────────────────────────────────────────────────────────────────
    # After running train_sft.py:
    !python scripts/train_dpo.py \\
        --config configs/training/dpo.yaml \\
        --model-config configs/model/qwen2_5_7b.yaml \\
        --sft-adapter experiments/sft_runs/run_001/final_adapter \\
        --override training.per_device_train_batch_size=1 \\
        --override training.gradient_accumulation_steps=16 \\
        --override dpo.max_prompt_length=512 \\
        --override dpo.max_length=1024 \\
        --wandb-key YOUR_KEY

────────────────────────────────────────────────────────────────────────
RUNPOD / VAST.AI (A100 40GB)
────────────────────────────────────────────────────────────────────────
    python scripts/train_dpo.py \\
        --config configs/training/dpo.yaml \\
        --model-config configs/model/qwen2_5_7b.yaml \\
        --sft-adapter experiments/sft_runs/run_001/final_adapter \\
        --wandb-key YOUR_KEY

Expected time on A100: ~1.5 hours for 1 epoch on 6K preference pairs.

────────────────────────────────────────────────────────────────────────
EXPECTED IMPROVEMENT OVER SFT
────────────────────────────────────────────────────────────────────────
    Hallucination:   SFT ~5.5%  → DPO ~2.8%   (-2.7pp)
    Instruction:     SFT ~93.5% → DPO ~96.0%  (+2.5pp)
    Format valid:    SFT ~97.5% → DPO ~98.5%  (+1.0pp)
    Schema:          SFT ~91.0% → DPO ~94.0%  (+3.0pp)
    Align. score:    SFT ~86.2% → DPO ~89.5%  (+3.3pp)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
# pyrefly: ignore [missing-import]
from omegaconf import DictConfig, OmegaConf

# pyrefly: ignore [missing-import]
from src.data.preprocessor import build_dpo_datasets
# pyrefly: ignore [missing-import]
from src.data.schemas import ModelStage, SFTExample
# pyrefly: ignore [missing-import]
from src.evaluation.benchmarks import EvaluationPipeline
# pyrefly: ignore [missing-import]
from src.evaluation.comparator import ModelComparator
# pyrefly: ignore [missing-import]
from src.models.lora_config import build_lora_config, prepare_model_for_training
# pyrefly: ignore [missing-import]
from src.models.loader import load_base_model, load_peft_model
# pyrefly: ignore [missing-import]
from src.models.parameter_counter import lora_efficiency_report, print_parameter_table
# pyrefly: ignore [missing-import]
from src.training.dpo_trainer import (
    build_dpo_datasets_from_disk,
    build_dpo_trainer,
)
# pyrefly: ignore [missing-import]
from src.training.utils import (
    find_resume_checkpoint,
    log_vram_usage,
    set_training_environment,
)
# pyrefly: ignore [missing-import]
from src.utils.config_loader import apply_overrides, load_config
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
        description="Phase 6 — DPO Alignment Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/training/dpo.yaml",
    )
    p.add_argument(
        "--model-config",
        type=str,
        default="configs/model/qwen2_5_7b.yaml",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override base model name",
    )
    p.add_argument(
        "--sft-adapter",
        type=str,
        default=None,
        help="Path to SFT adapter directory from Phase 5. "
             "If not provided, trains DPO on base model (less effective).",
    )
    p.add_argument(
        "--dpo-train-data",
        type=str,
        default="data/processed/dpo_train.jsonl",
        help="Path to DPO training JSONL file",
    )
    p.add_argument(
        "--dpo-val-data",
        type=str,
        default="data/processed/dpo_validation.jsonl",
        help="Path to DPO validation JSONL file",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override. E.g. --override dpo.beta=0.05",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="30-step test run for pipeline validation",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip post-training evaluation",
    )
    p.add_argument(
        "--explicit-ref-model",
        action="store_true",
        help="Load an explicit reference model instead of using implicit PEFT reference. "
             "Uses 2× VRAM. Only use if implicit reference gives poor results.",
    )
    p.add_argument(
        "--wandb-key",
        type=str,
        default=None,
    )
    return p.parse_args()


def setup_wandb(key: Optional[str], cfg: DictConfig) -> None:
    if cfg.training.get("report_to", "none") == "none":
        return
    try:
        # pyrefly: ignore [missing-import]
        import wandb, os
        if key:
            os.environ["WANDB_API_KEY"] = key
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "llm-alignment-system"),
            name=cfg.training.get("run_name", "dpo-training"),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["dpo", "alignment", cfg.model.name.split("/")[-1]],
            resume="allow",
        )
        logger = logging.getLogger(__name__)
        logger.info(f"[W&B] Run: {wandb.run.url}")
    except ImportError:
        pass
    except Exception as e:
        logging.getLogger(__name__).warning(f"W&B init failed: {e}")


def load_starting_model(
    cfg: DictConfig,
    sft_adapter_path: Optional[str],
    apply_fresh_lora: bool = True,
):
    """
    Load the model that will be trained with DPO.

    Strategy:
        If sft_adapter_path is provided:
            1. Load base model (quantized on GPU, float32 on CPU)
            2. Load SFT adapter on top (this is our SFT checkpoint)
            3. Apply FRESH LoRA adapters for DPO training
               (the SFT adapter weights act as the implicit reference)

        If no sft_adapter_path:
            1. Load base model
            2. Apply LoRA adapters directly
            DPO then trains policy relative to base model — less effective
            but works when SFT adapter is not available.

    Why fresh LoRA for DPO on top of SFT:
        The DPO loss measures drift from π_ref (the SFT policy).
        If we continued training the same SFT LoRA weights, we could not
        distinguish "how much did this step change from SFT?" from
        "how much did this step change from base?".
        Fresh LoRA weights initialised at zero give a clean reference point:
        ΔW_DPO = 0 at the start → π_θ = π_SFT at step 0 → clean KL baseline.
    """
    logger = logging.getLogger(__name__)

    is_quantized = cfg.quantization.get("load_in_4bit", False) and torch.cuda.is_available()

    logger.info("[DPO Loader] Loading base model...")
    model, tokenizer = load_base_model(cfg)
    log_vram_usage("after_base_model_load")

    if sft_adapter_path and Path(sft_adapter_path).exists():
        logger.info(f"[DPO Loader] Loading SFT adapter: {sft_adapter_path}")
        model = load_peft_model(model, sft_adapter_path, is_trainable=False)

        # Merge SFT adapter into the model weights so the base
        # (now = base + SFT adaptation) serves as the DPO reference
        if hasattr(model, "merge_and_unload"):
            logger.info(
                "[DPO Loader] Merging SFT adapter into base weights. "
                "The merged model becomes the DPO reference policy (π_ref)."
            )
            model = model.merge_and_unload()
        else:
            logger.info("[DPO Loader] SFT adapter loaded (not merged — using PEFT implicit ref)")
        log_vram_usage("after_sft_adapter_load")
    else:
        logger.warning(
            "[DPO Loader] No SFT adapter provided. "
            "DPO will train relative to the base model. "
            "Results will be weaker than DPO on top of SFT."
        )

    if apply_fresh_lora:
        logger.info("[DPO Loader] Applying fresh LoRA adapters for DPO training...")
        model = prepare_model_for_training(
            model, cfg, is_quantized=is_quantized
        )
        log_vram_usage("after_dpo_lora_injection")

    return model, tokenizer


def load_dpo_data(
    dpo_train_path: str,
    dpo_val_path: str,
    tokenizer=None,
    sft_examples_path: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> tuple:
    """
    Load DPO preference pairs from disk.

    Primary: load from pre-generated dpo_train.jsonl / dpo_validation.jsonl
    Fallback: generate pairs on-the-fly from SFT examples using preprocessor

    Args:
        dpo_train_path: Path to DPO training JSONL.
        dpo_val_path:   Path to DPO validation JSONL.
        tokenizer:      Required for fallback generation.
        sft_examples_path: Fallback SFT examples path.
        max_pairs:      Cap total pairs for smoke tests.

    Returns:
        (train_dataset, eval_dataset) tuple.
    """
    logger = logging.getLogger(__name__)

    if Path(dpo_train_path).exists():
        dpo_datasets = build_dpo_datasets_from_disk(dpo_train_path, dpo_val_path)
    elif sft_examples_path and Path(sft_examples_path).exists() and tokenizer:
        logger.info(
            f"DPO JSONL not found at {dpo_train_path}. "
            "Generating pairs from SFT examples..."
        )
        raw = read_jsonl_all(sft_examples_path)
        sft_examples = []
        for record in raw:
            try:
                sft_examples.append(SFTExample.from_dict(record))
            except Exception:
                continue
        dpo_datasets = build_dpo_datasets(sft_examples, tokenizer)
    else:
        logger.error(
            f"No DPO data found at {dpo_train_path} and no fallback available.\n"
            "Run: python scripts/generate_dataset.py to create DPO pairs."
        )
        sys.exit(1)

    train_ds = dpo_datasets["train"]
    eval_ds = dpo_datasets["validation"]

    if max_pairs:
        train_ds = train_ds.select(range(min(max_pairs, len(train_ds))))
        eval_pairs = max(10, max_pairs // 10)
        eval_ds = eval_ds.select(range(min(eval_pairs, len(eval_ds))))

    logger.info(
        f"[DPO Data] Final: train={len(train_ds):,}, eval={len(eval_ds):,}"
    )
    return train_ds, eval_ds


def log_dpo_metrics(trainer_state, logger) -> None:
    """
    Log DPO-specific metrics from trainer state.

    DPO logs additional metrics beyond the standard loss:
        rewards/chosen:    Mean log-ratio for chosen responses (should increase)
        rewards/rejected:  Mean log-ratio for rejected responses (should decrease)
        rewards/margins:   chosen - rejected (the alignment gap, should increase)
        rewards/accuracies: Fraction of pairs where chosen > rejected
    """
    if not trainer_state.log_history:
        return

    last_eval = [
        log for log in trainer_state.log_history
        if "eval_loss" in log
    ]
    if not last_eval:
        return

    final = last_eval[-1]
    metrics_to_show = {
        k: v for k, v in final.items()
        if any(kw in k for kw in [
            "eval_loss", "rewards/", "logps/", "logits/"
        ])
    }
    if metrics_to_show:
        log_dict(logger, metrics_to_show, "Final DPO eval metrics")


def main() -> None:
    args = parse_args()

    logger = setup_logging(
        level="INFO",
        log_dir="outputs/logs/training",
        run_name="train_dpo",
    )
    set_training_environment()

    # ── Load config ───────────────────────────────────────────────────────────
    log_section(logger, "Phase 6 — DPO Alignment Training")
    cfg = load_config(args.config)
    model_cfg = load_config(args.model_config)
    cfg = OmegaConf.merge(model_cfg, cfg)

    if args.override:
        cfg = apply_overrides(cfg, args.override)
    if args.model:
        cfg.model.name = args.model

    if args.smoke_test:
        cfg.training.max_steps = 30
        cfg.training.num_train_epochs = 1
        cfg.training.eval_steps = 15
        cfg.training.save_steps = 30
        cfg.training.logging_steps = 5
        cfg.training.report_to = "none"
        cfg.dpo.max_prompt_length = 256
        cfg.dpo.max_length = 512
        logger.info("[SmokeTest] max_steps=30, max_length=512")

    log_dict(
        logger,
        {
            "base_model":     cfg.model.name,
            "sft_adapter":    args.sft_adapter or "none",
            "beta":           cfg.dpo.beta,
            "loss_type":      cfg.dpo.get("loss_type", "sigmoid"),
            "lr":             cfg.training.learning_rate,
            "batch":          cfg.training.per_device_train_batch_size,
            "grad_accum":     cfg.training.gradient_accumulation_steps,
            "effective_batch": (
                cfg.training.per_device_train_batch_size
                * cfg.training.gradient_accumulation_steps
            ),
            "epochs":         cfg.training.num_train_epochs,
            "max_steps":      cfg.training.get("max_steps", -1),
            "max_length":     cfg.dpo.get("max_length", 2048),
            "output_dir":     cfg.training.output_dir,
        },
        "DPO Configuration",
    )

    # ── Setup ─────────────────────────────────────────────────────────────────
    seed = cfg.training.get("seed", 42)
    set_seed(seed)
    log_system_info()
    setup_wandb(args.wandb_key, cfg)
    ensure_dir(cfg.training.output_dir)
    ensure_dir("experiments/benchmark_results")
    ensure_dir("metrics")

    # ── Load model ────────────────────────────────────────────────────────────
    log_section(logger, "Loading model for DPO")
    log_vram_usage("start")

    try:
        model, tokenizer = load_starting_model(
            cfg=cfg,
            sft_adapter_path=args.sft_adapter,
            apply_fresh_lora=True,
        )
    except Exception as e:
        logger.error(
            f"Model load failed: {e}\n\n"
            "For local testing:\n"
            "  python scripts/train_dpo.py \\\n"
            "    --config configs/training/dpo_local.yaml \\\n"
            "    --model-config configs/model/qwen2_5_7b.yaml \\\n"
            "    --model Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "    --smoke-test"
        )
        sys.exit(1)

    print_parameter_table(model)
    efficiency = lora_efficiency_report(model)
    log_dict(logger, efficiency, "DPO LoRA efficiency")

    # ── Reference model ───────────────────────────────────────────────────────
    ref_model = None
    if args.explicit_ref_model:
        log_section(logger, "Loading explicit reference model")
        logger.info(
            "Loading an explicit reference model doubles VRAM usage.\n"
            "Only use this if the implicit PEFT reference gives poor results."
        )
        try:
            # pyrefly: ignore [missing-import]
            from src.models.loader import load_model_for_inference
            ref_base, _ = load_model_for_inference(cfg.model.name)
            if args.sft_adapter and Path(args.sft_adapter).exists():
                ref_model = load_peft_model(
                    ref_base, args.sft_adapter, is_trainable=False
                )
            else:
                ref_model = ref_base
            ref_model.eval()
            logger.info("[DPO] Explicit reference model loaded")
            log_vram_usage("after_ref_model_load")
        except Exception as e:
            logger.warning(
                f"Explicit reference model load failed: {e}\n"
                "Falling back to implicit PEFT reference (ref_model=None)."
            )
            ref_model = None

    # ── Load DPO data ─────────────────────────────────────────────────────────
    log_section(logger, "Loading DPO preference pairs")
    max_pairs = 200 if args.smoke_test else None
    train_ds, eval_ds = load_dpo_data(
        dpo_train_path=args.dpo_train_data,
        dpo_val_path=args.dpo_val_data,
        tokenizer=tokenizer,
        sft_examples_path="data/processed/train.jsonl",
        max_pairs=max_pairs,
    )

    # Log W&B data stats
    try:
        # pyrefly: ignore [missing-import]
        import wandb
        if wandb.run:
            wandb.config.update({
                "dpo_train_pairs":    len(train_ds),
                "dpo_eval_pairs":     len(eval_ds),
                "sft_adapter":        args.sft_adapter or "none",
                "explicit_ref_model": args.explicit_ref_model,
                "trainable_params_M": efficiency["trainable_params_M"],
                "efficiency_ratio":   efficiency["efficiency_ratio"],
            })
    except ImportError:
        pass

    # ── Build trainer ─────────────────────────────────────────────────────────
    log_section(logger, "Assembling DPOTrainer")
    trainer = build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        cfg=cfg,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    log_section(logger, "Starting DPO training")

    resume_from = None
    if args.resume:
        resume_from = find_resume_checkpoint(cfg.training.output_dir)
        if resume_from:
            logger.info(f"Resuming from: {resume_from}")

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        logger.info("DPO training interrupted. Saving checkpoint...")
        trainer.save_model(
            str(Path(cfg.training.output_dir) / "interrupted_adapter")
        )
        sys.exit(0)
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "CUDA OOM during DPO training.\n"
            "DPO processes pairs (2× sequences per batch).\n"
            "Solutions:\n"
            "  1. --override training.per_device_train_batch_size=1\n"
            "  2. --override dpo.max_length=1024\n"
            "  3. --override dpo.max_prompt_length=512\n"
            "  4. Use gradient_accumulation_steps instead of larger batch"
        )
        sys.exit(1)

    # ── Save final DPO adapter ────────────────────────────────────────────────
    log_section(logger, "Saving DPO adapter")
    adapter_path = Path(cfg.training.output_dir) / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    write_json(
        OmegaConf.to_container(cfg, resolve=True),
        str(adapter_path / "dpo_training_config.json"),
    )

    # pyrefly: ignore [missing-import]
    from src.utils.file_utils import get_dir_size_mb
    adapter_size = get_dir_size_mb(str(adapter_path))
    logger.info(f"DPO adapter saved → {adapter_path} ({adapter_size:.1f} MB)")

    # ── Training summary ──────────────────────────────────────────────────────
    metrics = train_result.metrics
    train_summary = {
        "train_loss":      round(metrics.get("train_loss", 0), 4),
        "train_runtime_h": round(metrics.get("train_runtime", 0) / 3600, 2),
        "best_eval_loss":  round(trainer.state.best_metric or 0, 4),
        "adapter_size_mb": round(adapter_size, 1),
        "adapter_path":    str(adapter_path),
        "beta":            cfg.dpo.beta,
        "loss_type":       cfg.dpo.get("loss_type", "sigmoid"),
    }
    log_dict(logger, train_summary, "DPO training summary")
    log_dpo_metrics(trainer.state, logger)

    try:
        # pyrefly: ignore [missing-import]
        import wandb
        if wandb.run:
            wandb.log({f"dpo_final/{k}": v for k, v in train_summary.items()})
    except ImportError:
        pass

    # ── Post-training evaluation ───────────────────────────────────────────────
    if not args.no_eval:
        log_section(logger, "Post-DPO evaluation")

        test_examples: list[SFTExample] = []
        test_path = Path("data/processed/test.jsonl")
        if test_path.exists():
            for record in read_jsonl_all(str(test_path)):
                try:
                    test_examples.append(SFTExample.from_dict(record))
                except Exception:
                    continue

        if test_examples:
            model.eval()
            eval_max = 100 if args.smoke_test else None

            eval_pipeline = EvaluationPipeline(
                model=model,
                tokenizer=tokenizer,
                model_stage=ModelStage.DPO,
                model_id=str(adapter_path),
            )
            dpo_result = eval_pipeline.run(
                test_examples=test_examples,
                output_dir="experiments/benchmark_results",
                max_examples=eval_max,
                save_predictions=True,
            )

            save_metrics(
                metrics=dpo_result.to_dict(),
                path="metrics/dpo_metrics.json",
                run_name=cfg.training.get("run_name", "dpo"),
            )

            # Full comparison: Base → SFT → DPO
            comparator = ModelComparator()
            comparator.add(dpo_result)
            for stage, path in [
                ("base", "experiments/benchmark_results/benchmark_base.json"),
                ("sft",  "experiments/benchmark_results/benchmark_sft.json"),
            ]:
                if Path(path).exists():
                    comparator.add_from_file(path)

            log_section(logger, "Final Comparison: Base → SFT → DPO")
            comparator.print_comparison_table()
            comparator.save_comparison(
                "experiments/benchmark_results/full_comparison.json"
            )
            comparator.update_history("metrics/benchmark_history.csv")

            try:
                # pyrefly: ignore [missing-import]
                import wandb
                if wandb.run:
                    wandb.log({
                        "dpo_eval/format_valid":        dpo_result.format_valid,
                        "dpo_eval/schema_compliant":    dpo_result.schema_compliant,
                        "dpo_eval/instruction_followed": dpo_result.instruction_followed,
                        "dpo_eval/hallucination_rate":  dpo_result.hallucination_rate,
                        "dpo_eval/avg_field_f1":        dpo_result.avg_field_f1,
                        "dpo_eval/avg_alignment_score": dpo_result.avg_alignment_score,
                    })
            except ImportError:
                pass
        else:
            logger.warning("No test examples found — skipping evaluation")

    try:
        # pyrefly: ignore [missing-import]
        import wandb
        if wandb.run:
            wandb.finish()
    except ImportError:
        pass

    log_section(logger, "Phase 6 Complete")
    logger.info(
        f"DPO adapter: {adapter_path}\n"
        f"Metrics:     metrics/dpo_metrics.json\n"
        f"Comparison:  experiments/benchmark_results/full_comparison.json\n\n"
        f"Next — Phase 7 (Inference + Deployment):\n"
        f"  Merge DPO adapter for vLLM:\n"
        f"    python scripts/export_model.py \\\n"
        f"      --base-model {cfg.model.name} \\\n"
        f"      --adapter {adapter_path} \\\n"
        f"      --output outputs/models/dpo/merged\n\n"
        f"  Launch inference API:\n"
        f"    python scripts/serve.py \\\n"
        f"      --model outputs/models/dpo/merged"
    )


if __name__ == "__main__":
    main()