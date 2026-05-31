# pyrefly: ignore [missing-import]
"""
Evaluate a specific model checkpoint (SFT or DPO adapter).

Used after training to measure improvement over baseline.

Usage:
    # Evaluate SFT adapter
    python scripts/evaluate.py \\
        --model-stage sft \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --adapter experiments/sft_runs/run_001/final_adapter \\
        --max-examples 200

    # Evaluate DPO adapter
    python scripts/evaluate.py \\
        --model-stage dpo \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --adapter experiments/dpo_runs/run_001/final_adapter \\
        --max-examples 200

    # Evaluate merged model (for vLLM deployment)
    python scripts/evaluate.py \\
        --model-stage sft \\
        --merged-model outputs/models/sft/merged \\
        --max-examples 200
"""
from __future__ import annotations

import argparse
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
from src.models.loader import load_base_model, load_model_for_inference, load_peft_model
# pyrefly: ignore [missing-import]
from src.utils.file_utils import ensure_dir, read_jsonl_all, save_metrics
# pyrefly: ignore [missing-import]
from src.utils.logging import log_section, setup_logging
# pyrefly: ignore [missing-import]
from src.utils.reproducibility import set_seed


def load_test_examples(path: str, max_examples: int) -> list[SFTExample]:
    raw = read_jsonl_all(path)
    examples = []
    for record in raw:
        try:
            examples.append(SFTExample.from_dict(record))
        except Exception:
            continue
    if max_examples > 0:
        examples = examples[:max_examples]
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model against the test set"
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        required=True,
        choices=["sft", "dpo"],
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name (used with --adapter)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to PEFT adapter directory",
    )
    parser.add_argument(
        "--merged-model",
        type=str,
        default=None,
        help="Path to merged model directory (no adapter needed)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging(
        level="INFO",
        log_dir="outputs/logs/training",
        run_name=f"eval_{args.model_stage}",
    )
    set_seed(args.seed)

    stage = ModelStage(args.model_stage)
    output_dir = args.output_dir or f"experiments/benchmark_results"
    ensure_dir(output_dir)
    ensure_dir("metrics")

    log_section(logger, f"Evaluation — {args.model_stage.upper()}")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.merged_model:
        logger.info(f"Loading merged model: {args.merged_model}")
        model, tokenizer = load_model_for_inference(
            args.merged_model, dtype=args.dtype
        )
        model_id = args.merged_model
    elif args.adapter:
        logger.info(f"Loading base + adapter: {args.adapter}")
        import torch
        # pyrefly: ignore [missing-import]
        from omegaconf import OmegaConf
        # pyrefly: ignore [missing-import]
        from src.utils.config_loader import load_config

        # Load base model without quantization for clean inference
        model, tokenizer = load_model_for_inference(
            args.base_model, dtype=args.dtype
        )
        model = load_peft_model(model, args.adapter, is_trainable=False)
        model_id = args.adapter
    else:
        logger.error("Provide either --adapter or --merged-model")
        sys.exit(1)

    # ── Load test examples ─────────────────────────────────────────────────────
    test_examples = load_test_examples(args.test_data, args.max_examples)
    logger.info(f"Evaluating on {len(test_examples)} examples")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    pipeline = EvaluationPipeline(
        model=model,
        tokenizer=tokenizer,
        model_stage=stage,
        model_id=model_id,
    )

    result = pipeline.run(
        test_examples=test_examples,
        output_dir=output_dir,
        max_examples=None,
        save_predictions=True,
    )

    # ── Save metrics ───────────────────────────────────────────────────────────
    metrics_path = f"metrics/{args.model_stage}_metrics.json"
    save_metrics(
        metrics=result.to_dict(),
        path=metrics_path,
        run_name=model_id,
    )
    logger.info(f"Metrics saved → {metrics_path}")

    # ── Update history and compare ────────────────────────────────────────────
    comparator = ModelComparator()
    comparator.add(result)

    base_path = "experiments/benchmark_results/benchmark_base.json"
    if Path(base_path).exists():
        comparator.add_from_file(base_path)
        comparator.print_comparison_table()

    comparator.update_history("metrics/benchmark_history.csv")

    log_section(logger, f"Evaluation Complete — {args.model_stage.upper()}")
    logger.info(result.summary())


if __name__ == "__main__":
    main()