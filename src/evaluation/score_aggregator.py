"""
Aggregates EvaluationResult lists into BenchmarkResult.

Handles:
    - Global metric means across all examples
    - Per-task breakdown (invoice / support_ticket / tool_call)
    - Per-difficulty breakdown (easy / medium / hard / adversarial)
    - Failure mode frequency counts
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np

# pyrefly: ignore [missing-import]
from src.data.schemas import (
    BenchmarkResult,
    DifficultyLevel,
    DifficultyMetrics,
    EvaluationResult,
    FailureMode,
    ModelStage,
    TaskMetrics,
    TaskType,
)
# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ScoreAggregator:
    """
    Converts a list of EvaluationResult objects to a BenchmarkResult.

    All aggregations are explicit numpy operations so behaviour is
    transparent and debuggable. No hidden weighting.
    """

    def aggregate(
        self,
        results: list[EvaluationResult],
        model_id: str,
        model_stage: ModelStage,
    ) -> BenchmarkResult:
        """
        Aggregate evaluation results into a BenchmarkResult.

        Args:
            results:     List of EvaluationResult from MetricsComputer.
            model_id:    Model identifier string.
            model_stage: Pipeline stage enum value.

        Returns:
            Fully populated BenchmarkResult.
        """
        if not results:
            logger.warning("[Aggregator] Empty results list — returning empty BenchmarkResult")
            return BenchmarkResult(model_id=model_id, model_stage=model_stage)

        benchmark = BenchmarkResult(
            model_id=model_id,
            model_stage=model_stage,
            n_examples=len(results),
        )

        # ── Global metrics ────────────────────────────────────────────────────
        benchmark.exact_match = float(np.mean([r.exact_match for r in results]))
        benchmark.format_valid = float(np.mean([r.format_valid for r in results]))
        benchmark.schema_compliant = float(np.mean([r.schema_compliant for r in results]))
        benchmark.instruction_followed = float(np.mean([r.instruction_followed for r in results]))
        benchmark.hallucination_rate = float(np.mean([r.hallucination_detected for r in results]))
        benchmark.avg_bleu = float(np.mean([r.bleu for r in results]))
        benchmark.avg_rouge_l = float(np.mean([r.rouge_l for r in results]))
        benchmark.avg_field_f1 = float(np.mean([r.field_f1 for r in results]))
        benchmark.avg_alignment_score = float(np.mean([r.alignment_score for r in results]))

        # ── Round all floats to 4 decimal places ──────────────────────────────
        for attr in [
            "exact_match", "format_valid", "schema_compliant",
            "instruction_followed", "hallucination_rate",
            "avg_bleu", "avg_rouge_l", "avg_field_f1", "avg_alignment_score",
        ]:
            setattr(benchmark, attr, round(getattr(benchmark, attr), 4))

        # ── Per-task breakdown ────────────────────────────────────────────────
        by_task: dict[str, list[EvaluationResult]] = defaultdict(list)
        for r in results:
            by_task[r.task_type.value].append(r)

        benchmark.n_tasks = len(by_task)
        for task_name, task_results in by_task.items():
            tm = TaskMetrics(
                task_type=TaskType(task_name),
                n_examples=len(task_results),
                exact_match=round(float(np.mean([r.exact_match for r in task_results])), 4),
                format_valid=round(float(np.mean([r.format_valid for r in task_results])), 4),
                schema_compliant=round(float(np.mean([r.schema_compliant for r in task_results])), 4),
                instruction_followed=round(float(np.mean([r.instruction_followed for r in task_results])), 4),
                hallucination_rate=round(float(np.mean([r.hallucination_detected for r in task_results])), 4),
                avg_bleu=round(float(np.mean([r.bleu for r in task_results])), 4),
                avg_rouge_l=round(float(np.mean([r.rouge_l for r in task_results])), 4),
                avg_field_f1=round(float(np.mean([r.field_f1 for r in task_results])), 4),
                avg_alignment_score=round(float(np.mean([r.alignment_score for r in task_results])), 4),
                avg_latency_ms=round(float(np.mean([r.latency_ms for r in task_results])), 1),
            )
            benchmark.by_task[task_name] = tm

        # ── Per-difficulty breakdown ───────────────────────────────────────────
        by_diff: dict[str, list[EvaluationResult]] = defaultdict(list)
        for r in results:
            by_diff[r.difficulty.value].append(r)

        for diff_name, diff_results in by_diff.items():
            dm = DifficultyMetrics(
                difficulty=DifficultyLevel(diff_name),
                n_examples=len(diff_results),
                format_valid=round(float(np.mean([r.format_valid for r in diff_results])), 4),
                instruction_followed=round(float(np.mean([r.instruction_followed for r in diff_results])), 4),
                hallucination_rate=round(float(np.mean([r.hallucination_detected for r in diff_results])), 4),
                avg_alignment_score=round(float(np.mean([r.alignment_score for r in diff_results])), 4),
            )
            benchmark.by_difficulty[diff_name] = dm

        # ── Failure mode counts ───────────────────────────────────────────────
        mode_counter: Counter = Counter()
        for r in results:
            for mode in r.failure_modes:
                mode_counter[mode.value] += 1
        benchmark.failure_mode_counts = dict(mode_counter)

        logger.info(
            f"[Aggregator] {model_stage.value} | "
            f"n={len(results)} | "
            f"format={benchmark.format_valid*100:.1f}% | "
            f"schema={benchmark.schema_compliant*100:.1f}% | "
            f"align={benchmark.avg_alignment_score*100:.1f}%"
        )
        return benchmark