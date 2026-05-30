"""
Phase 4 — Complete baseline evaluation pipeline.

Runs all four prompt engineering strategies on the base model,
compares them, and saves the best result as the official baseline
that SFT and DPO are measured against.

Pipeline:
    1. Load test examples from data/processed/test.jsonl
    2. Load base model (no fine-tuning, no LoRA)
    3. For each strategy (zero_shot, few_shot, cot, few_shot_cot):
        a. Build prompts using the strategy
        b. Run greedy inference on each test example
        c. Compute all metrics (format_valid, field_f1, etc.)
        d. Aggregate into BenchmarkResult
        e. Save predictions and results to disk
    4. Compare all strategies — print table
    5. Save best strategy result as metrics/baseline_metrics.json
    6. Update metrics/benchmark_history.csv

Local execution (Windows AMD — no GPU):
    python scripts/run_baseline.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --max-examples 20 \\
        --strategies zero_shot few_shot \\
        --dtype float32

    Expected time: ~2-5 minutes on CPU with 0.5B model, 20 examples.

Cloud execution (Colab A100):
    python scripts/run_baseline.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --max-examples 0 \\
        --strategies zero_shot few_shot chain_of_thought few_shot_cot \\
        --dtype bfloat16

    Expected time: ~15-25 minutes on A100 with 7B model, full test set.

Colab quick start:
    !git clone https://github.com/your-repo/llm-alignment-system
    %cd llm-alignment-system
    !pip install -e . -q
    !python scripts/generate_dataset.py --no-public
    !python scripts/run_baseline.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --max-examples 100 \\
        --dtype bfloat16

RunPod / Vast.ai:
    Same as Colab but in a terminal. Use a PyTorch 2.3 template with CUDA 12.1.
    Recommended: A100 40GB ($1.50/hr RunPod) or RTX 4090 ($0.50/hr Vast.ai)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schemas import (
    BenchmarkResult,
    DifficultyLevel,
    EvaluationResult,
    FailureMode,
    ModelStage,
    SFTExample,
    TaskType,
)
from src.evaluation.baseline_strategies import get_all_strategies, get_strategy
from src.evaluation.benchmarks import MetricsComputer
from src.evaluation.comparator import ModelComparator
from src.evaluation.latency import LatencyProfiler
from src.evaluation.score_aggregator import ScoreAggregator
from src.models.loader import load_model_for_inference
from src.utils.file_utils import (
    append_jsonl,
    ensure_dir,
    read_jsonl_all,
    save_metrics,
    write_json,
)
from src.utils.logging import log_dict, log_section, setup_logging
from src.utils.reproducibility import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_test_examples(path: str, max_examples: int) -> list[SFTExample]:
    """Load SFTExample objects from the test JSONL file."""
    logger = logging.getLogger(__name__)

    if not Path(path).exists():
        logger.error(
            f"Test file not found: {path}\n"
            "Run: python scripts/generate_dataset.py"
        )
        sys.exit(1)

    raw = read_jsonl_all(path)
    examples, skipped = [], 0

    for record in raw:
        try:
            ex = SFTExample.from_dict(record)
            examples.append(ex)
        except Exception as e:
            logger.debug(f"Skipping record: {e}")
            skipped += 1

    if max_examples > 0:
        examples = examples[:max_examples]

    logger.info(
        f"Loaded {len(examples)} test examples "
        f"({skipped} skipped)"
    )
    return examples


@torch.inference_mode()
def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.1,
) -> tuple[str, float, int, int]:
    """
    Run greedy inference and return (generated_text, latency_ms, in_tok, out_tok).

    Greedy decoding (do_sample=False, temperature ignored):
        - Fully deterministic given the same model weights
        - Essential for reproducible evaluation metrics
        - If you use sampling, the same model gives different outputs
          each run — metric differences become noise, not signal

    torch.inference_mode():
        More efficient than torch.no_grad():
        - Disables gradient tracking AND autograd version counters
        - ~10% faster for inference workloads
        - Cannot compute gradients (correct for evaluation)
    """
    device = next(model.parameters()).device

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536,
        padding=False,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    start = time.perf_counter()

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if device.type == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_ids = model.generate(**inputs, **gen_kwargs)
    else:
        output_ids = model.generate(**inputs, **gen_kwargs)

    latency_ms = (time.perf_counter() - start) * 1000
    new_ids = output_ids[0][input_len:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return generated, latency_ms, input_len, len(new_ids)


def evaluate_strategy(
    strategy,
    model,
    tokenizer,
    test_examples: list[SFTExample],
    output_dir: str,
    max_new_tokens: int = 512,
) -> BenchmarkResult:
    """
    Run one strategy across all test examples and return BenchmarkResult.

    Saves per-example predictions and results to JSONL files in output_dir.
    Results are saved incrementally so partial runs are recoverable.
    """
    logger = logging.getLogger(__name__)
    strategy_name = strategy.name
    log_section(logger, f"Strategy: {strategy_name}")

    ensure_dir(output_dir)
    preds_path = Path(output_dir) / f"predictions_{strategy_name}.jsonl"
    results_path = Path(output_dir) / f"results_{strategy_name}.jsonl"

    metrics_computer = MetricsComputer()
    latency_profiler = LatencyProfiler()
    all_results: list[EvaluationResult] = []

    model.eval()

    for example in tqdm(test_examples, desc=strategy_name):
        try:
            reference = example.target_text
        except ValueError:
            continue

        source_text = example.input_text
        prediction_id = str(uuid4())

        # Build prompt with this strategy
        try:
            prompt = strategy.build_prompt(example, tokenizer)
        except Exception as e:
            logger.warning(f"Prompt build failed for {example.example_id}: {e}")
            continue

        # Inference
        try:
            lat_start = latency_profiler.start()
            generated, latency_ms, in_tok, out_tok = run_inference(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )
            latency_profiler.end(
                lat_start,
                request_id=prediction_id,
                input_tokens=in_tok,
                output_tokens=out_tok,
                model_stage="base",
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "GPU OOM during inference. "
                    "Reduce --max-new-tokens or use a smaller model."
                )
                torch.cuda.empty_cache()
                continue
            logger.warning(f"Inference failed: {e}")
            generated, latency_ms, in_tok, out_tok = "", 0.0, 0, 0

        # Save raw prediction
        append_jsonl(
            {
                "prediction_id": prediction_id,
                "example_id": example.example_id,
                "strategy": strategy_name,
                "task_type": example.task_type.value,
                "difficulty": example.difficulty.value,
                "prompt_length": len(prompt),
                "generated_text": generated,
                "reference_text": reference,
                "source_text": source_text[:300],
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": latency_ms,
            },
            str(preds_path),
        )

        # Compute metrics
        eval_result = metrics_computer.compute(
            prediction_text=generated,
            reference_text=reference,
            task_type=example.task_type,
            source_text=source_text,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model_stage=ModelStage.BASE,
            example_id=example.example_id,
            prediction_id=prediction_id,
        )
        eval_result.difficulty = example.difficulty
        all_results.append(eval_result)
        append_jsonl(eval_result.to_dict(), str(results_path))

    # Aggregate
    aggregator = ScoreAggregator()
    benchmark = aggregator.aggregate(
        results=all_results,
        model_id="base_model",
        model_stage=ModelStage.BASE,
    )

    # Attach latency stats
    lat_summary = latency_profiler.summary()
    benchmark.avg_latency_ms = lat_summary.get("mean_latency_ms", 0.0)
    benchmark.p95_latency_ms = lat_summary.get("p95_latency_ms", 0.0)
    benchmark.avg_tokens_per_second = lat_summary.get("mean_tokens_per_second", 0.0)

    # Save benchmark JSON
    bench_path = Path(output_dir) / f"benchmark_{strategy_name}.json"
    result_dict = benchmark.to_dict()
    result_dict["strategy"] = strategy_name
    write_json(result_dict, str(bench_path))

    logger.info(benchmark.summary())
    logger.info(
        f"Saved: predictions={preds_path.name}, "
        f"results={results_path.name}, "
        f"benchmark={bench_path.name}"
    )
    return benchmark


def pick_best_strategy(
    strategy_results: dict[str, BenchmarkResult],
) -> tuple[str, BenchmarkResult]:
    """
    Select the best strategy by composite alignment score.

    Uses avg_alignment_score as the primary metric because it weights:
        - format compliance (25%)
        - instruction following (25%)
        - hallucination avoidance (20%)
        - field F1 (15%)
        - ROUGE-L (10%)
        - exact match (5%)

    This is more robust than picking by any single metric.
    """
    best_name, best_result = max(
        strategy_results.items(),
        key=lambda kv: kv[1].avg_alignment_score,
    )
    return best_name, best_result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 — Baseline evaluation with prompt engineering strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or local path",
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
        help="Directory for all outputs",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "few_shot", "chain_of_thought", "few_shot_cot"],
        choices=["zero_shot", "few_shot", "chain_of_thought", "few_shot_cot"],
        help="Which strategies to run",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help="Max test examples. 0 = all. Use 20 for quick local test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype. Use float32 for CPU-only machines.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
    ensure_dir("experiments/benchmark_results")

    log_section(logger, "Phase 4 — Baseline Pipeline")
    log_dict(
        logger,
        {
            "model":        args.model,
            "strategies":   args.strategies,
            "max_examples": args.max_examples or "all",
            "max_new_tokens": args.max_new_tokens,
            "dtype":        args.dtype,
            "output_dir":   args.output_dir,
        },
        "Configuration",
    )

    # ── Load test data ────────────────────────────────────────────────────────
    test_examples = load_test_examples(
        args.test_data,
        max_examples=args.max_examples,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    log_section(logger, "Loading base model")
    logger.info(
        f"Loading: {args.model}\n"
        f"This may take 30-90 seconds depending on model size and hardware."
    )

    try:
        model, tokenizer = load_model_for_inference(
            model_name_or_path=args.model,
            dtype=args.dtype,
        )
    except Exception as e:
        logger.error(
            f"Model load failed: {e}\n\n"
            "Troubleshooting:\n"
            "  CPU / no GPU:  python scripts/run_baseline.py "
            "--model Qwen/Qwen2.5-0.5B-Instruct --dtype float32\n"
            "  No internet:   python scripts/run_baseline.py "
            "--model distilgpt2 --dtype float32 --max-examples 5\n"
            "  GPU OOM:       Enable QLoRA in loader or use smaller model"
        )
        sys.exit(1)

    # ── Run each strategy ─────────────────────────────────────────────────────
    log_section(logger, "Running strategies")
    strategy_results: dict[str, BenchmarkResult] = {}

    for strategy_name in args.strategies:
        strategy = get_strategy(strategy_name)
        result = evaluate_strategy(
            strategy=strategy,
            model=model,
            tokenizer=tokenizer,
            test_examples=test_examples,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
        )
        strategy_results[strategy_name] = result

    if not strategy_results:
        logger.error("No strategies completed successfully.")
        sys.exit(1)

    # ── Strategy comparison table ─────────────────────────────────────────────
    log_section(logger, "Strategy Comparison")
    _print_strategy_table(strategy_results, logger)

    # ── Pick best strategy ────────────────────────────────────────────────────
    best_name, best_result = pick_best_strategy(strategy_results)
    logger.info(
        f"\nBest strategy: {best_name.upper()}\n"
        f"  Alignment score: {best_result.avg_alignment_score*100:.1f}%\n"
        f"  Format valid:    {best_result.format_valid*100:.1f}%\n"
        f"  Field F1:        {best_result.avg_field_f1*100:.1f}%\n"
        f"  Hallucination:   {best_result.hallucination_rate*100:.1f}%"
    )

    # ── Save official baseline ────────────────────────────────────────────────
    baseline_data = best_result.to_dict()
    baseline_data["best_strategy"] = best_name
    baseline_data["all_strategy_scores"] = {
        name: {
            "alignment_score": round(r.avg_alignment_score * 100, 1),
            "format_valid":    round(r.format_valid * 100, 1),
            "field_f1":        round(r.avg_field_f1 * 100, 1),
            "hallucination":   round(r.hallucination_rate * 100, 1),
        }
        for name, r in strategy_results.items()
    }

    save_metrics(
        metrics=baseline_data,
        path="metrics/baseline_metrics.json",
        run_name=f"baseline-{args.model.split('/')[-1]}",
    )

    # Copy best benchmark to the shared benchmark_results dir for comparator
    write_json(
        baseline_data,
        "experiments/benchmark_results/benchmark_base.json",
    )

    # ── Update benchmark history CSV ──────────────────────────────────────────
    comparator = ModelComparator()
    comparator.add(best_result)
    comparator.update_history("metrics/benchmark_history.csv")

    # ── Per-task and per-difficulty breakdown ─────────────────────────────────
    log_section(logger, f"Best Strategy ({best_name}) — Breakdown")
    _print_per_task_breakdown(best_result, logger)
    _print_per_difficulty_breakdown(best_result, logger)
    _print_failure_modes(best_result, logger)

    # ── Next steps ────────────────────────────────────────────────────────────
    log_section(logger, "Phase 4 Complete")
    logger.info(
        f"Official baseline saved → metrics/baseline_metrics.json\n"
        f"Benchmark result saved → experiments/benchmark_results/benchmark_base.json\n"
        f"History updated       → metrics/benchmark_history.csv\n\n"
        f"Next steps:\n"
        f"  Phase 5: SFT training\n"
        f"    python scripts/train_sft.py\n\n"
        f"  After training, evaluate and compare:\n"
        f"    python scripts/evaluate.py --model-stage sft "
        f"--adapter experiments/sft_runs/run_001/final_adapter\n\n"
        f"  Final comparison:\n"
        f"    python scripts/benchmark.py"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_strategy_table(
    results: dict[str, BenchmarkResult],
    logger: logging.Logger,
) -> None:
    metrics = [
        ("Format Valid",      "format_valid"),
        ("Schema Compliant",  "schema_compliant"),
        ("Instr. Followed",   "instruction_followed"),
        ("Hallucination ↓",   "hallucination_rate"),
        ("Field F1",          "avg_field_f1"),
        ("ROUGE-L",           "avg_rouge_l"),
        ("Align. Score",      "avg_alignment_score"),
        ("Latency (ms) ↓",    "avg_latency_ms"),
    ]

    names = list(results.keys())
    col_w = max(12, max(len(n) for n in names) + 2)
    bar = "═" * (24 + col_w * len(names))

    lines = [f"\n{bar}"]
    header = f"  {'Metric':<22}"
    for name in names:
        header += f"{name.upper():>{col_w}}"
    lines.append(header)
    lines.append(f"{'─' * (24 + col_w * len(names))}")

    for label, attr in metrics:
        row = f"  {label:<22}"
        for name in names:
            val = getattr(results[name], attr, 0.0)
            if "latency" in attr:
                row += f"{val:>{col_w}.0f}ms"[: col_w]
                row += " " * max(0, col_w - len(f"{val:.0f}ms"))
            else:
                row += f"{val * 100:>{col_w}.1f}%"
        lines.append(row)

    lines.append(f"{'─' * (24 + col_w * len(names))}")
    lines.append(
        f"  {'Examples':<22}"
        + "".join(f"{results[n].n_examples:>{col_w}}" for n in names)
    )
    lines.append(bar + "\n")

    for line in lines:
        logger.info(line)


def _print_per_task_breakdown(
    result: BenchmarkResult,
    logger: logging.Logger,
) -> None:
    if not result.by_task:
        return
    logger.info("Per-task breakdown:")
    for task, metrics in result.by_task.items():
        logger.info(
            f"  {task:<30} "
            f"format={metrics.format_valid*100:.1f}%  "
            f"align={metrics.avg_alignment_score*100:.1f}%  "
            f"n={metrics.n_examples}"
        )


def _print_per_difficulty_breakdown(
    result: BenchmarkResult,
    logger: logging.Logger,
) -> None:
    if not result.by_difficulty:
        return
    logger.info("Per-difficulty breakdown:")
    for diff, metrics in result.by_difficulty.items():
        logger.info(
            f"  {diff:<15} "
            f"format={metrics.format_valid*100:.1f}%  "
            f"halluc={metrics.hallucination_rate*100:.1f}%  "
            f"n={metrics.n_examples}"
        )


def _print_failure_modes(
    result: BenchmarkResult,
    logger: logging.Logger,
) -> None:
    if not result.failure_mode_counts:
        return
    logger.info("Top failure modes:")
    sorted_modes = sorted(
        result.failure_mode_counts.items(),
        key=lambda x: -x[1],
    )
    for mode, count in sorted_modes[:6]:
        pct = 100 * count / max(result.n_examples, 1)
        logger.info(f"  {mode:<30} {count:>5} ({pct:.1f}%)")


if __name__ == "__main__":
    main()