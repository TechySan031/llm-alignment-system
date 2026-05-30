"""
Core metrics computation and evaluation pipeline.

Two classes do all the work:

MetricsComputer:
    Stateless. Takes one (prediction_text, reference_text, source_text) triple
    and returns a fully populated EvaluationResult. No model, no tokenizer,
    no GPU. Can be used in unit tests without any model loaded.

EvaluationPipeline:
    Stateful. Holds a loaded model and tokenizer. Iterates over test examples,
    calls the model for each, measures latency, calls MetricsComputer,
    aggregates results into a BenchmarkResult, and saves predictions to disk.

Design decision — saving predictions separately:
    Raw model outputs are saved to disk alongside evaluation results.
    This means you can re-run metric computation with updated metrics
    (e.g. adding a new metric after evaluation) without re-running expensive
    inference. This is a standard MLOps pattern.

Greedy decoding for evaluation:
    temperature=0, do_sample=False gives fully deterministic output.
    This is essential for fair comparison — if you use sampling, the same
    model gives different outputs on different evaluation runs, making
    metric differences meaningless noise rather than real signal.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional
from uuid import uuid4

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

# pyrefly: ignore [missing-import]
from src.data.schemas import (
    BenchmarkResult,
    DifficultyLevel,
    DifficultyMetrics,
    EvaluationResult,
    FailureMode,
    ModelPrediction,
    ModelStage,
    SFTExample,
    TaskMetrics,
    TaskType,
)
# pyrefly: ignore [missing-import]
from src.evaluation.hallucination import HallucinationDetector
# pyrefly: ignore [missing-import]
from src.evaluation.json_validator import JSONValidator
# pyrefly: ignore [missing-import]
from src.evaluation.latency import LatencyProfiler
# pyrefly: ignore [missing-import]
from src.evaluation.tool_call_metrics import ToolCallMetrics
# pyrefly: ignore [missing-import]
from src.models.tokenizer_loader import format_messages_prompt
# pyrefly: ignore [missing-import]
from src.utils.file_utils import append_jsonl, ensure_dir, write_json
# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger, log_section

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lexical similarity utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_set(text: str) -> set[str]:
    return set(_normalise(text).split())


def compute_token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 score between two strings.

    Used for field-level comparison in structured extraction:
        precision = |pred_tokens ∩ ref_tokens| / |pred_tokens|
        recall    = |pred_tokens ∩ ref_tokens| / |ref_tokens|
        f1        = 2 * precision * recall / (precision + recall)

    Returns 1.0 for empty strings that match (both empty).
    Returns 0.0 for empty prediction against non-empty reference.
    """
    pred_tokens = _token_set(prediction)
    ref_tokens = _token_set(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = len(pred_tokens & ref_tokens)
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L F1 score using longest common subsequence.

    ROUGE-L captures sentence-level structure — words must appear
    in order (though not necessarily contiguously). More appropriate
    than ROUGE-1/2 for structured output evaluation where order matters.
    """
    pred_tokens = _normalise(prediction).split()
    ref_tokens = _normalise(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Dynamic programming LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    if lcs == 0:
        return 0.0

    precision = lcs / m
    recall = lcs / n
    return round(2 * precision * recall / (precision + recall), 4)


def compute_field_f1(pred_data: dict, ref_data: dict) -> float:
    """
    Mean token-level F1 across all string fields in a JSON dict.

    For each field present in the reference:
        - Find the same field in the prediction
        - Compute token F1 between the two string values
        - Average across all fields

    Fields with None values in both pred and ref score 1.0.
    Fields missing from prediction score 0.0.
    """
    if not isinstance(pred_data, dict) or not isinstance(ref_data, dict):
        return 0.0

    def flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten(v, key))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    out.update(flatten(item, f"{key}[{i}]"))
            else:
                out[key] = v
        return out

    pred_flat = flatten(pred_data)
    ref_flat = flatten(ref_data)

    f1_scores = []
    for key, ref_val in ref_flat.items():
        pred_val = pred_flat.get(key)

        if ref_val is None and pred_val is None:
            f1_scores.append(1.0)
            continue
        if ref_val is None or pred_val is None:
            f1_scores.append(0.0)
            continue

        ref_str = str(ref_val)
        pred_str = str(pred_val)

        if isinstance(ref_val, (int, float)) and isinstance(pred_val, (int, float)):
            # Numeric: exact match within 2% tolerance
            if ref_val == 0:
                f1_scores.append(1.0 if pred_val == 0 else 0.0)
            else:
                rel_err = abs(float(pred_val) - float(ref_val)) / abs(float(ref_val))
                f1_scores.append(1.0 if rel_err <= 0.02 else 0.0)
        elif isinstance(ref_val, bool):
            f1_scores.append(1.0 if pred_val == ref_val else 0.0)
        else:
            f1_scores.append(compute_token_f1(pred_str, ref_str))

    return round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MetricsComputer
# ─────────────────────────────────────────────────────────────────────────────

class MetricsComputer:
    """
    Stateless metrics computation for one (prediction, reference) pair.

    All methods are pure functions of their inputs.
    No model, no tokenizer, no GPU required.
    Can be instantiated and used in unit tests with no setup.

    Metrics computed:
        exact_match:           Normalised string equality
        format_valid:          Prediction is parseable JSON / follows format
        schema_compliant:      Prediction matches Pydantic schema (task-specific)
        instruction_followed:  Proxy: format_valid AND no extra preamble/postamble
        hallucination_detected: Any hallucination type detected
        bleu:                  BLEU-1 score (unigram precision with brevity penalty)
        rouge_l:               ROUGE-L F1
        field_f1:              Mean token F1 across JSON fields
        tool_metrics:          Tool call accuracy (tool_call tasks only)
    """

    def __init__(self):
        self._json_validator = JSONValidator()
        self._hallucination_detector = HallucinationDetector()
        self._tool_metrics = ToolCallMetrics()

    def compute(
        self,
        prediction_text: str,
        reference_text: str,
        task_type: TaskType,
        source_text: str = "",
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_stage: ModelStage = ModelStage.BASE,
        example_id: Optional[str] = None,
        prediction_id: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Compute all evaluation metrics for one prediction.

        Args:
            prediction_text: Raw text generated by the model.
            reference_text:  Ground truth expected output.
            task_type:       Task type for task-specific metrics.
            source_text:     Original input document (for hallucination grounding).
            latency_ms:      Inference latency in milliseconds.
            input_tokens:    Number of prompt tokens (for TPS computation).
            output_tokens:   Number of generated tokens.
            model_stage:     Which pipeline stage generated this prediction.
            example_id:      Link to source SFTExample.
            prediction_id:   Link to ModelPrediction record.

        Returns:
            Fully populated EvaluationResult.
        """
        result = EvaluationResult(
            model_stage=model_stage,
            task_type=task_type,
            difficulty=DifficultyLevel.MEDIUM,  # Updated by caller if known
            latency_ms=latency_ms,
            tokens_per_second=round(
                (input_tokens + output_tokens) / max(latency_ms / 1000, 1e-6), 1
            ),
            example_id=example_id,
            prediction_id=prediction_id,
        )

        # ── JSON validation ───────────────────────────────────────────────────
        validation = self._json_validator.validate(prediction_text, reference_text)
        result.format_valid = validation["json_valid"]
        result.schema_compliant = validation["schema_compliant"]

        pred_data = validation["pred_data"] or {}
        ref_data = validation["ref_data"] or {}

        # Instruction followed: format valid + no preamble/postamble
        result.instruction_followed = (
            result.format_valid
            and self._no_prose_wrapper(prediction_text)
        )

        # ── Exact match ───────────────────────────────────────────────────────
        result.exact_match = self._normalised_exact_match(
            prediction_text, reference_text
        )

        # ── Lexical similarity ────────────────────────────────────────────────
        result.rouge_l = compute_rouge_l(prediction_text, reference_text)
        result.bleu = self._compute_bleu(prediction_text, reference_text)

        # ── Field F1 (JSON tasks) ─────────────────────────────────────────────
        if result.format_valid and ref_data:
            result.field_f1 = compute_field_f1(pred_data, ref_data)
        else:
            result.field_f1 = 0.0

        # ── Hallucination ─────────────────────────────────────────────────────
        if result.format_valid and ref_data:
            halluc_report = self._hallucination_detector.analyze(
                pred_data, ref_data, source_text
            )
            result.hallucination_detected = halluc_report.hallucination_detected
        else:
            result.hallucination_detected = False

        # ── Failure mode classification ───────────────────────────────────────
        result.failure_modes = self._classify_failure_modes(result, prediction_text)

        return result

    def _normalised_exact_match(self, prediction: str, reference: str) -> bool:
        """
        Normalised exact match: strip whitespace, lowercase, then compare.

        For JSON: compare parsed dicts (ignoring key order and whitespace).
        """
        # Try JSON-aware comparison
        try:
            pred_obj = json.loads(prediction.strip())
            ref_obj = json.loads(reference.strip())
            return self._normalise_obj(pred_obj) == self._normalise_obj(ref_obj)
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: string comparison
        return _normalise(prediction) == _normalise(reference)

    def _normalise_obj(self, obj):
        """Recursively normalise a JSON object for comparison."""
        if isinstance(obj, dict):
            return {k: self._normalise_obj(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [self._normalise_obj(i) for i in obj]
        if isinstance(obj, str):
            return obj.lower().strip()
        if isinstance(obj, float):
            return round(obj, 2)
        return obj

    def _no_prose_wrapper(self, text: str) -> bool:
        """
        Check that the model did not wrap JSON in prose.

        Returns False if the text has significant content before the first {
        or after the last }, indicating a verbose wrapper failure mode.
        """
        stripped = text.strip()
        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")

        if first_brace == -1:
            return False

        preamble = stripped[:first_brace].strip()
        postamble = stripped[last_brace + 1:].strip() if last_brace != -1 else ""

        # Allow very short preamble (e.g. model might output newline before {)
        return len(preamble) <= 5 and len(postamble) <= 5

    def _compute_bleu(self, prediction: str, reference: str) -> float:
        """
        BLEU-1 score (unigram precision with brevity penalty).

        Full BLEU uses n-grams up to n=4 and a corpus-level brevity penalty.
        We use BLEU-1 at the example level as a rough lexical overlap proxy.
        Use ROUGE-L as the primary continuous metric.
        """
        pred_tokens = _normalise(prediction).split()
        ref_tokens = _normalise(reference).split()

        if not pred_tokens:
            return 0.0
        if not ref_tokens:
            return 0.0

        ref_token_counts: dict[str, int] = {}
        for t in ref_tokens:
            ref_token_counts[t] = ref_token_counts.get(t, 0) + 1

        matches = 0
        used: dict[str, int] = {}
        for token in pred_tokens:
            if ref_token_counts.get(token, 0) > used.get(token, 0):
                matches += 1
                used[token] = used.get(token, 0) + 1

        precision = matches / len(pred_tokens)

        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1))
        import math
        if len(pred_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
        else:
            bp = 1.0

        return round(bp * precision, 4)

    def _classify_failure_modes(
        self,
        result: EvaluationResult,
        prediction_text: str,
    ) -> list[FailureMode]:
        """
        Classify what went wrong for failed predictions.

        Returning a list allows multiple concurrent failure modes.
        Used for breakdown analysis: "what fraction of DPO failures
        are format errors vs hallucination vs truncation?"
        """
        if result.exact_match:
            return [FailureMode.NONE]

        modes = []

        if not result.format_valid:
            # Distinguish truncation from format error
            stripped = prediction_text.strip()
            if stripped.startswith("{") and not stripped.endswith("}"):
                modes.append(FailureMode.TRUNCATED_OUTPUT)
            else:
                modes.append(FailureMode.FORMAT_ERROR)

        if result.format_valid and not result.instruction_followed:
            modes.append(FailureMode.INSTRUCTION_IGNORED)

        if result.hallucination_detected:
            modes.append(FailureMode.FIELD_HALLUCINATION)

        if result.format_valid and not result.schema_compliant:
            modes.append(FailureMode.SCHEMA_VIOLATION)

        # Refusal detection (model said it can't help)
        refusal_phrases = [
            "i cannot", "i can't", "i am unable", "i'm unable",
            "as an ai", "i don't have access", "i'm sorry, but",
        ]
        if any(p in prediction_text.lower() for p in refusal_phrases):
            modes.append(FailureMode.REFUSAL)

        return modes if modes else [FailureMode.NONE]


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationPipeline
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationPipeline:
    """
    Orchestrates full evaluation: inference → metrics → aggregation → saving.

    Usage:
        pipeline = EvaluationPipeline(model, tokenizer, model_stage=ModelStage.SFT)
        result = pipeline.run(
            test_examples=test_examples,
            output_dir="experiments/benchmark_results/sft",
        )
        print(result.summary())
    """

    # Greedy generation config for reproducible evaluation
    _GENERATION_CONFIG = {
        "max_new_tokens": 512,
        "temperature": 1.0,      # Ignored when do_sample=False
        "do_sample": False,       # Greedy decoding
        "repetition_penalty": 1.1,
    }

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_stage: ModelStage,
        model_id: str = "unknown",
        generation_config: Optional[dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_stage = model_stage
        self.model_id = model_id
        self.generation_config = generation_config or self._GENERATION_CONFIG.copy()

        self._metrics_computer = MetricsComputer()
        self._latency_profiler = LatencyProfiler()
        self._tool_metrics = ToolCallMetrics()

        # Set pad_token_id in generation config
        self.generation_config["pad_token_id"] = tokenizer.pad_token_id
        self.generation_config["eos_token_id"] = tokenizer.eos_token_id

    @torch.inference_mode()
    def _generate_one(self, prompt: str) -> tuple[str, float, int, int]:
        """
        Run one inference call and return (generated_text, latency_ms, input_tok, output_tok).

        torch.inference_mode() is more efficient than torch.no_grad():
            - Disables gradient tracking AND version counter updates
            - ~10% faster for inference workloads
            - Cannot be used for training (no gradients computed at all)

        For CPU inference: dtype autocast is skipped (no benefit on CPU).
        For GPU inference: bfloat16 autocast reduces memory and improves speed.
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,  # Reserve 512 tokens for generation
            padding=False,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]
        start = time.perf_counter()

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    **self.generation_config,
                )
        else:
            # CPU: no autocast
            output_ids = self.model.generate(
                **inputs,
                **self.generation_config,
            )

        latency_ms = (time.perf_counter() - start) * 1000

        # Decode only the newly generated tokens (not the prompt)
        new_ids = output_ids[0][input_len:]
        generated_text = self.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        ).strip()

        output_len = len(new_ids)
        return generated_text, latency_ms, input_len, output_len

    def _build_prompt(self, example: SFTExample) -> str:
        """
        Build the inference prompt from an SFTExample.

        Uses the prompt_only turns (system + user) with add_generation_prompt=True.
        This mirrors exactly how DPO pair prompts are built.
        """
        prompt_messages = [
            {"role": t.role.value, "content": t.content}
            for t in example.conversation.prompt_only
        ]
        return format_messages_prompt(
            prompt_messages,
            self.tokenizer,
            add_generation_prompt=True,
        )

    def run(
        self,
        test_examples: list[SFTExample],
        output_dir: str = "experiments/benchmark_results",
        max_examples: Optional[int] = None,
        save_predictions: bool = True,
    ) -> BenchmarkResult:
        """
        Run evaluation on a list of SFTExample objects.

        Args:
            test_examples:    List of SFTExample from the test split.
            output_dir:       Where to save predictions and results.
            max_examples:     Cap for quick development runs.
                              None evaluates the full test set.
            save_predictions: If True, save raw predictions to JSONL
                              alongside evaluation results.

        Returns:
            BenchmarkResult with all aggregated metrics.
        """
        ensure_dir(output_dir)
        stage_name = self.model_stage.value

        if max_examples is not None:
            test_examples = test_examples[:max_examples]

        log_section(logger, f"Evaluation: {stage_name} | {len(test_examples)} examples")

        predictions_path = Path(output_dir) / f"predictions_{stage_name}.jsonl"
        results_path = Path(output_dir) / f"results_{stage_name}.jsonl"

        self.model.eval()
        all_results: list[EvaluationResult] = []
        self._latency_profiler.reset()

        for example in tqdm(test_examples, desc=f"Evaluating [{stage_name}]"):
            # Get reference response
            try:
                reference_text = example.target_text
            except ValueError:
                logger.debug(f"Skipping {example.example_id} — no assistant turn")
                continue

            source_text = example.input_text
            prompt = self._build_prompt(example)
            prediction_id = str(uuid4())

            # Inference
            try:
                generated_text, latency_ms, in_tok, out_tok = self._generate_one(prompt)
            except Exception as e:
                logger.warning(f"Inference failed for {example.example_id}: {e}")
                generated_text = ""
                latency_ms = 0.0
                in_tok, out_tok = 0, 0

            # Save prediction to disk
            if save_predictions:
                pred_record = {
                    "prediction_id": prediction_id,
                    "example_id": example.example_id,
                    "model_stage": stage_name,
                    "model_id": self.model_id,
                    "task_type": example.task_type.value,
                    "difficulty": example.difficulty.value,
                    "source": example.source,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "reference_text": reference_text,
                    "source_text": source_text[:500],  # Truncate for storage
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "latency_ms": latency_ms,
                }
                append_jsonl(pred_record, str(predictions_path))

            # Compute metrics
            eval_result = self._metrics_computer.compute(
                prediction_text=generated_text,
                reference_text=reference_text,
                task_type=example.task_type,
                source_text=source_text,
                latency_ms=latency_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                model_stage=self.model_stage,
                example_id=example.example_id,
                prediction_id=prediction_id,
            )
            eval_result.difficulty = example.difficulty
            all_results.append(eval_result)

            # Save result to disk
            if save_predictions:
                append_jsonl(eval_result.to_dict(), str(results_path))

        # Aggregate
        # pyrefly: ignore [missing-import]
        from src.evaluation.score_aggregator import ScoreAggregator
        aggregator = ScoreAggregator()
        benchmark = aggregator.aggregate(
            results=all_results,
            model_id=self.model_id,
            model_stage=self.model_stage,
        )

        # Add latency stats
        latency_summary = self._latency_profiler.summary()
        benchmark.avg_latency_ms = latency_summary.get("mean_latency_ms", 0.0)
        benchmark.p95_latency_ms = latency_summary.get("p95_latency_ms", 0.0)
        benchmark.avg_tokens_per_second = latency_summary.get("mean_tokens_per_second", 0.0)

        # Save benchmark result
        result_file = Path(output_dir) / f"benchmark_{stage_name}.json"
        write_json(benchmark.to_dict(), str(result_file))
        logger.info(f"Benchmark saved → {result_file}")
        logger.info(benchmark.summary())

        return benchmark