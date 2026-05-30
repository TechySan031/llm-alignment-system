"""
Phase 3 execution script: evaluate the base model before any training.

Establishes the before-training benchmark that SFT and DPO results
are compared against. Run this ONCE before training starts.

Local execution (Windows AMD — no GPU):
    Uses distilgpt2 or Qwen2.5-0.5B-Instruct for testing.
    Inference is slow on CPU (~2-10 seconds per example).
    Use --max-examples 20 for a quick smoke test locally.

Cloud execution (Colab / RunPod with GPU):
    Uses Qwen2.5-7B-Instruct in bfloat16.
    Inference is ~300-500ms per example on A100.
    Evaluate on the full test split (--max-examples 0 = all).

Usage:
    # Local quick test (no download, uses distilgpt2)
    python scripts/run_baseline.py --model distilgpt2 --max-examples 10 --no-public

    # Local full test with small Qwen model
    python scripts/run_baseline.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --max-examples 50

    # Cloud full evaluation
    python scripts/run_baseline.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --max-examples 0

Expected outputs:
    experiments/baseline_runs/benchmark_base.json
    experiments/baseline_runs/predictions_base.jsonl
    experiments/baseline_runs/results_base.jsonl
    metrics/baseline_metrics.json

Expected metrics (Qwen2.5-7B-Instruct zero-shot, from experience):
    Format Valid:        ~75-85%   (knows JSON but sometimes adds prose)
    Schema Compliant:    ~45-60%   (misses schema details)
    Instruction Followed: ~60-70%  (sometimes ignores format constraints)
    Hallucination Rate:  ~20-30%   (adds confidence scores, timestamps)
    Field F1:            ~55-70%
    Alignment Score:     ~50-65%
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from src.data.schemas import ModelStage, SFTExample
# pyrefly: ignore [missing-import]
from src.evaluation.benchmarks import EvaluationPipeline
# pyrefly: ignore [missing-import]
from src.evaluation.comparator import ModelComparator
# pyrefly: ignore [missing-import]
from src.models.loader import load_model_for_inference
# pyrefly: ignore [missing-import]
from src.utils.file_utils import ensure_dir, read_jsonl_all, save_metrics, write_json
# pyrefly: ignore [missing-import]
from src.utils.logging import log_section, setup_logging
# pyrefly: ignore [missing-import]
from src.utils.reproducibility import set_seed


def load_test_examples(test_jsonl_path: str) -> list[SFTExample]:
    """
    Load test examples from the JSONL file produced by generate_dataset.py.

    Skips malformed records with a warning rather than crashing.
    """
    raw_records = read_jsonl_all(test_jsonl_path)
    examples: list[SFTExample] = []
    skipped = 0

    for record in raw_records:
        try:
            ex = SFTExample.from_dict(record)
            examples.append(ex)
        except Exception as e:
            logging.getLogger(__name__).debug(
                f"Skipping malformed record: {e}"
            )
            skipped += 1

    logging.getLogger(__name__).info(
        f"Loaded {len(examples)} test examples "
        f"({skipped} skipped)"
    )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation before training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to evaluate (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baseline_runs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help="Max test examples to evaluate. 0 = all. (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (use float32 for CPU-only machines)",
    )
    args = parser.parse_args()

    logger = setup_logging(
        level="INFO",
        log_dir="outputs/logs/training",
        run_name="baseline_eval",
    )
    set_seed(args.seed)

    ensure_dir(args.output_dir)
    ensure_dir("metrics")

    log_section(logger, "Phase 3 — Baseline Evaluation")
    logger.info(f"Model:       {args.model}")
    logger.info(f"Test data:   {args.test_data}")
    logger.info(f"Max examples: {args.max_examples or 'all'}")

    # ── Load test examples ────────────────────────────────────────────────────
    if not Path(args.test_data).exists():
        logger.error(
            f"Test data not found: {args.test_data}\n"
            "Run: python scripts/generate_dataset.py first"
        )
        sys.exit(1)

    test_examples = load_test_examples(args.test_data)
    if not test_examples:
        logger.error("No test examples loaded. Check the test JSONL file.")
        sys.exit(1)

    max_ex = args.max_examples if args.max_examples > 0 else len(test_examples)
    logger.info(f"Evaluating on {min(max_ex, len(test_examples))} examples")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model}")
    try:
        model, tokenizer = load_model_for_inference(
            model_name_or_path=args.model,
            dtype=args.dtype,
        )
    except Exception as e:
        logger.error(
            f"Failed to load model: {e}\n\n"
            "For local testing without internet, use distilgpt2:\n"
            "  python scripts/run_baseline.py --model distilgpt2 --max-examples 5\n\n"
            "For small Qwen model:\n"
            "  python scripts/run_baseline.py --model Qwen/Qwen2.5-0.5B-Instruct --max-examples 20"
        )
        sys.exit(1)

    # ── Run evaluation ────────────────────────────────────────────────────────
    pipeline = EvaluationPipeline(
        model=model,
        tokenizer=tokenizer,
        model_stage=ModelStage.BASE,
        model_id=args.model,
    )

    benchmark_result = pipeline.run(
        test_examples=test_examples,
        output_dir=args.output_dir,
        max_examples=max_ex if args.max_examples > 0 else None,
        save_predictions=True,
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    save_metrics(
        metrics=benchmark_result.to_dict(),
        path="metrics/baseline_metrics.json",
        run_name=f"baseline-{args.model.split('/')[-1]}",
    )

    # ── Print results ─────────────────────────────────────────────────────────
    log_section(logger, "Baseline Results")
    logger.info(benchmark_result.summary())

    logger.info(
        "\nPer-task breakdown:\n"
        + "\n".join(
            f"  {task}: format={m.format_valid*100:.1f}% | "
            f"align={m.avg_alignment_score*100:.1f}% | "
            f"n={m.n_examples}"
            for task, m in benchmark_result.by_task.items()
        )
    )

    logger.info(
        "\nPer-difficulty breakdown:\n"
        + "\n".join(
            f"  {diff}: format={m.format_valid*100:.1f}% | "
            f"halluc={m.hallucination_rate*100:.1f}% | "
            f"n={m.n_examples}"
            for diff, m in benchmark_result.by_difficulty.items()
        )
    )

    logger.info(
        f"\nFailure modes:\n"
        + "\n".join(
            f"  {mode}: {count}"
            for mode, count in sorted(
                benchmark_result.failure_mode_counts.items(),
                key=lambda x: -x[1],
            )
        )
    )

    # ── Update history ─────────────────────────────────────────────────────────
    comparator = ModelComparator()
    comparator.add(benchmark_result)
    comparator.update_history("metrics/benchmark_history.csv")

    logger.info(
        f"\nBaseline evaluation complete.\n"
        f"Results saved to: {args.output_dir}/\n"
        f"Metrics saved to: metrics/baseline_metrics.json\n\n"
        f"Next steps:\n"
        f"  1. Review benchmark_{ModelStage.BASE.value}.json\n"
        f"  2. Run SFT training: python scripts/train_sft.py\n"
        f"  3. Re-run evaluation with the SFT model\n"
        f"  4. Compare: python scripts/evaluate.py --compare"
    )


if __name__ == "__main__":
    main()