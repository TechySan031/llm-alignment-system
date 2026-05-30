"""
Multi-model comparison: Base → Prompt → SFT → DPO.

ModelComparator loads BenchmarkResult files saved by EvaluationPipeline
and produces:
    1. A comparison table (printed to console and saved to CSV)
    2. Delta metrics (improvement over baseline for each stage)
    3. Per-task and per-difficulty breakdowns
    4. The data structure used by visualization/benchmark_charts.py

This is the module that produces the "money table" for your portfolio —
the quantitative proof that your alignment pipeline improves the model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# pyrefly: ignore [missing-import]
from src.data.schemas import BenchmarkResult, ModelStage
# pyrefly: ignore [missing-import]
from src.utils.file_utils import read_json, write_json, write_csv, update_benchmark_history
# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger, log_section

logger = get_logger(__name__)

_STAGE_ORDER = [
    ModelStage.BASE,
    ModelStage.PROMPT_ENGINEERED,
    ModelStage.SFT,
    ModelStage.DPO,
]

_DISPLAY_METRICS = [
    ("format_valid",          "Format Valid",       True),
    ("schema_compliant",      "Schema Compliant",   True),
    ("instruction_followed",  "Instr. Followed",    True),
    ("hallucination_rate",    "Hallucination ↓",    False),  # Lower is better
    ("avg_field_f1",          "Field F1",           True),
    ("avg_rouge_l",           "ROUGE-L",            True),
    ("avg_alignment_score",   "Align. Score",       True),
    ("avg_latency_ms",        "Latency (ms) ↓",     False),
]


class ModelComparator:
    """
    Loads and compares BenchmarkResult objects across model stages.

    Usage:
        comparator = ModelComparator()
        comparator.add("experiments/benchmark_results/benchmark_base.json")
        comparator.add("experiments/benchmark_results/benchmark_sft.json")
        comparator.add("experiments/benchmark_results/benchmark_dpo.json")
        comparator.print_comparison_table()
        comparator.save_comparison("experiments/benchmark_results/comparison.json")
    """

    def __init__(self):
        self.results: dict[str, BenchmarkResult] = {}

    def add_from_file(self, path: str) -> None:
        """
        Load a BenchmarkResult from a JSON file saved by EvaluationPipeline.

        Args:
            path: Path to benchmark_*.json file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: {path}\n"
                "Run scripts/evaluate.py first."
            )
        data = read_json(path)
        stage_str = data.get("model_stage", "base")
        stage = ModelStage(stage_str)

        # Reconstruct BenchmarkResult
        result = BenchmarkResult(
            run_id=data.get("run_id", ""),
            model_id=data.get("model_id", "unknown"),
            model_stage=stage,
            n_examples=data.get("n_examples", 0),
            n_tasks=data.get("n_tasks", 0),
            exact_match=data.get("exact_match", 0.0),
            format_valid=data.get("format_valid", 0.0),
            schema_compliant=data.get("schema_compliant", 0.0),
            instruction_followed=data.get("instruction_followed", 0.0),
            hallucination_rate=data.get("hallucination_rate", 0.0),
            avg_bleu=data.get("avg_bleu", 0.0),
            avg_rouge_l=data.get("avg_rouge_l", 0.0),
            avg_field_f1=data.get("avg_field_f1", 0.0),
            avg_alignment_score=data.get("avg_alignment_score", 0.0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            p95_latency_ms=data.get("p95_latency_ms", 0.0),
            avg_tokens_per_second=data.get("avg_tokens_per_second", 0.0),
            failure_mode_counts=data.get("failure_mode_counts", {}),
        )
        self.results[stage_str] = result
        logger.info(f"[Comparator] Loaded: {stage_str} ({result.n_examples} examples)")

    def add(self, result: BenchmarkResult) -> None:
        """Add a BenchmarkResult object directly (from a live evaluation run)."""
        self.results[result.model_stage.value] = result

    def get_ordered_results(self) -> list[tuple[str, BenchmarkResult]]:
        """Return results in the canonical stage order."""
        ordered = []
        for stage in _STAGE_ORDER:
            key = stage.value
            if key in self.results:
                ordered.append((key, self.results[key]))
        # Add any stages not in the canonical order
        for key, result in self.results.items():
            if key not in [s.value for s in _STAGE_ORDER]:
                ordered.append((key, result))
        return ordered

    def print_comparison_table(self) -> None:
        """
        Print the full comparison table to stdout.

        Example output:
            ════════════════════════════════════════════════════════════════════
              Metric               Base     Prompt    SFT      DPO     Δ(SFT→DPO)
            ════════════════════════════════════════════════════════════════════
              Format Valid        52.3%    68.1%    91.4%    93.2%     +1.8pp
              Schema Compliant    41.0%    55.2%    88.7%    91.5%     +2.8pp
              Hallucination ↓     28.4%    19.3%     7.1%     4.8%     -2.3pp
              Align. Score        44.2%    58.7%    82.3%    86.1%     +3.8pp
              Latency (ms) ↓       412      398      421      445     +24ms
            ════════════════════════════════════════════════════════════════════
        """
        ordered = self.get_ordered_results()
        if not ordered:
            logger.warning("[Comparator] No results to compare")
            return

        stage_labels = [s.upper() for s, _ in ordered]
        col_width = max(10, max(len(l) for l in stage_labels) + 2)
        bar = "═" * (28 + col_width * len(ordered) + 12)

        print(f"\n{bar}")
        header = f"  {'Metric':<26}"
        for label in stage_labels:
            header += f"{label:>{col_width}}"
        if len(ordered) >= 2:
            header += "  Δ(last two)"
        print(header)
        print(f"{bar}")

        for attr, label, higher_is_better in _DISPLAY_METRICS:
            row = f"  {label:<26}"
            values = []
            for _, result in ordered:
                val = getattr(result, attr, 0.0)
                values.append(val)
                if "latency" in attr.lower():
                    row += f"{val:>{col_width}.0f}ms"[: col_width]
                    row += " " * max(0, col_width - len(f"{val:.0f}ms"))
                else:
                    row += f"{val * 100:>{col_width}.1f}%"

            # Delta between last two stages
            if len(values) >= 2:
                delta = values[-1] - values[-2]
                if "latency" in attr.lower():
                    sign = "+" if delta > 0 else ""
                    direction = "↑ worse" if delta > 0 else "↓ better"
                    print(f"{row}  {sign}{delta:.0f}ms ({direction})")
                else:
                    pp = delta * 100
                    sign = "+" if pp > 0 else ""
                    arrow = "↑" if higher_is_better == (pp > 0) else "↓"
                    print(f"{row}  {sign}{pp:.1f}pp {arrow}")
            else:
                print(row)

        print(f"{bar}\n")
        print(f"  Examples evaluated: {', '.join(str(r.n_examples) for _, r in ordered)}")
        print(f"{bar}\n")

    def compute_deltas(self, baseline_stage: str = "base") -> dict:
        """
        Compute improvement of each stage over the baseline.

        Args:
            baseline_stage: Stage to use as the baseline ("base" or "prompt_engineered").

        Returns:
            Dict of {stage: {metric: delta_pp}} for all non-baseline stages.
        """
        if baseline_stage not in self.results:
            logger.warning(
                f"[Comparator] Baseline stage '{baseline_stage}' not found. "
                f"Available: {list(self.results.keys())}"
            )
            return {}

        baseline = self.results[baseline_stage]
        deltas = {}

        for stage, result in self.results.items():
            if stage == baseline_stage:
                continue
            deltas[stage] = result.delta_vs(baseline)

        return deltas

    def save_comparison(
        self,
        output_path: str,
        baseline_stage: str = "base",
    ) -> None:
        """
        Save the full comparison to a JSON file.

        Saved file includes:
            - All benchmark results
            - Delta metrics vs baseline
            - Per-task breakdown comparison
            - Failure mode comparison
        """
        ordered = self.get_ordered_results()
        deltas = self.compute_deltas(baseline_stage)

        comparison = {
            "stages": [s for s, _ in ordered],
            "baseline_stage": baseline_stage,
            "results": {
                stage: result.to_dict()
                for stage, result in ordered
            },
            "deltas_vs_baseline": deltas,
            "summary": {
                stage: {
                    "format_valid_pct": round(result.format_valid * 100, 1),
                    "schema_compliant_pct": round(result.schema_compliant * 100, 1),
                    "instruction_followed_pct": round(result.instruction_followed * 100, 1),
                    "hallucination_rate_pct": round(result.hallucination_rate * 100, 1),
                    "avg_alignment_score_pct": round(result.avg_alignment_score * 100, 1),
                    "avg_latency_ms": result.avg_latency_ms,
                    "n_examples": result.n_examples,
                }
                for stage, result in ordered
            },
        }

        write_json(comparison, output_path)
        logger.info(f"[Comparator] Comparison saved → {output_path}")

    def update_history(
        self,
        csv_path: str = "metrics/benchmark_history.csv",
    ) -> None:
        """
        Append all current results to the benchmark history CSV.

        Used to track model improvement over multiple training runs.
        The CSV is the data source for the convergence plot notebook.
        """
        from datetime import datetime
        for stage, result in self.results.items():
            row = {
                "run_name": result.model_id,
                "model_stage": stage,
                "evaluated_at": datetime.utcnow().isoformat() + "Z",
                "format_valid": result.format_valid,
                "schema_compliant": result.schema_compliant,
                "instruction_followed": result.instruction_followed,
                "hallucination_rate": result.hallucination_rate,
                "avg_alignment_score": result.avg_alignment_score,
                "avg_bleu": result.avg_bleu,
                "avg_rouge_l": result.avg_rouge_l,
                "avg_field_f1": result.avg_field_f1,
                "avg_latency_ms": result.avg_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "avg_tokens_per_second": result.avg_tokens_per_second,
                "n_examples": result.n_examples,
            }
            update_benchmark_history(row, csv_path)
        logger.info(f"[Comparator] History updated → {csv_path}")