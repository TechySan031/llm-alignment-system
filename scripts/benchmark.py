"""
Cross-stage benchmark comparison: Base → SFT → DPO.

Loads saved benchmark JSON files from all pipeline stages and
produces the final comparison table + saves to metrics/.

Run after completing all training stages:
    python scripts/benchmark.py

Or specify individual benchmark files:
    python scripts/benchmark.py \\
        --base    experiments/baseline_runs/benchmark_base.json \\
        --sft     experiments/benchmark_results/benchmark_sft.json \\
        --dpo     experiments/benchmark_results/benchmark_dpo.json

Also usable mid-training to compare base vs SFT before DPO:
    python scripts/benchmark.py \\
        --base experiments/baseline_runs/benchmark_base.json \\
        --sft  experiments/benchmark_results/benchmark_sft.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from src.evaluation.comparator import ModelComparator
# pyrefly: ignore [missing-import]
from src.utils.file_utils import ensure_dir, write_json
# pyrefly: ignore [missing-import]
from src.utils.logging import log_section, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across pipeline stages"
    )
    parser.add_argument(
        "--base",
        type=str,
        default="experiments/benchmark_results/benchmark_base.json",
    )
    parser.add_argument(
        "--sft",
        type=str,
        default=None,
        help="Path to SFT benchmark JSON (optional)",
    )
    parser.add_argument(
        "--dpo",
        type=str,
        default=None,
        help="Path to DPO benchmark JSON (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/benchmark_results/full_comparison.json",
    )
    args = parser.parse_args()

    logger = setup_logging(
        level="INFO",
        log_dir="outputs/logs",
        run_name="benchmark_comparison",
    )
    ensure_dir("experiments/benchmark_results")
    ensure_dir("metrics")

    log_section(logger, "Benchmark Comparison")

    comparator = ModelComparator()
    loaded = []

    for stage, path in [("base", args.base), ("sft", args.sft), ("dpo", args.dpo)]:
        if path and Path(path).exists():
            comparator.add_from_file(path)
            loaded.append(stage)
            logger.info(f"Loaded: {stage} ({path})")
        elif path:
            logger.warning(f"File not found: {path} (skipping {stage})")

    if not loaded:
        logger.error(
            "No benchmark files found.\n"
            "Run the baseline first: python scripts/run_baseline.py"
        )
        sys.exit(1)

    comparator.print_comparison_table()

    if len(loaded) >= 2:
        deltas = comparator.compute_deltas(baseline_stage="base")
        logger.info("\nDeltas vs base:")
        for stage, stage_deltas in deltas.items():
            logger.info(f"  {stage.upper()}:")
            for metric, delta in sorted(stage_deltas.items()):
                if abs(delta) > 0.1:
                    sign = "+" if delta > 0 else ""
                    logger.info(f"    {metric:<40} {sign}{delta:.2f}")

    comparator.save_comparison(args.output, baseline_stage="base")
    comparator.update_history("metrics/benchmark_history.csv")

    logger.info(f"\nComparison saved → {args.output}")


if __name__ == "__main__":
    main()