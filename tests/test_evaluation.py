"""
Phase 3 unit tests — evaluation layer.

All tests run on CPU without loading any model.
MetricsComputer and JSONValidator are pure Python — no GPU required.

Run:
    pytest tests/test_evaluation.py -v
    pytest tests/test_evaluation.py -v -k "json"
"""
from __future__ import annotations

import json

import pytest

# pyrefly: ignore [missing-import]
from src.data.schemas import (
    DifficultyLevel,
    EvaluationResult,
    FailureMode,
    ModelStage,
    TaskType,
)
# pyrefly: ignore [missing-import]
from src.evaluation.benchmarks import (
    MetricsComputer,
    compute_field_f1,
    compute_rouge_l,
    compute_token_f1,
)
# pyrefly: ignore [missing-import]
from src.evaluation.comparator import ModelComparator
# pyrefly: ignore [missing-import]
from src.evaluation.hallucination import HallucinationDetector
# pyrefly: ignore [missing-import]
from src.evaluation.json_validator import (
    JSONValidator,
    extract_json_from_text,
)
# pyrefly: ignore [missing-import]
from src.evaluation.latency import LatencyMeasurement, LatencyProfiler
# pyrefly: ignore [missing-import]
from src.evaluation.score_aggregator import ScoreAggregator
# pyrefly: ignore [missing-import]
from src.evaluation.tool_call_metrics import ToolCallMetrics


# ─── JSON extraction tests ────────────────────────────────────────────────────

class TestExtractJsonFromText:
    def test_direct_valid_json(self):
        result = extract_json_from_text('{"key": "value"}')
        assert result.success
        assert result.data == {"key": "value"}
        assert result.strategy == "direct"

    def test_stripped_with_whitespace(self):
        result = extract_json_from_text('  {"key": "value"}  \n')
        assert result.success
        assert result.data["key"] == "value"

    def test_markdown_fence_explicit(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result.success
        assert result.data["key"] == "value"
        assert result.strategy == "fence_explicit"

    def test_markdown_fence_any(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result.success

    def test_preamble_extraction(self):
        text = 'Here is the extracted data:\n\n{"vendor": "ACME", "total": 100.0}'
        result = extract_json_from_text(text)
        assert result.success
        assert result.data["vendor"] == "ACME"
        assert result.strategy == "first_object"

    def test_postamble_extraction(self):
        text = '{"vendor": "ACME"}\n\nLet me know if you need clarification.'
        result = extract_json_from_text(text)
        assert result.success
        assert result.data["vendor"] == "ACME"

    def test_single_quotes_converted(self):
        text = "{'vendor': 'ACME', 'total': 100.0}"
        result = extract_json_from_text(text)
        assert result.success
        assert result.data["vendor"] == "ACME"

    def test_empty_string_fails(self):
        result = extract_json_from_text("")
        assert not result.success
        assert result.strategy == "none"

    def test_truncated_json_fails(self):
        result = extract_json_from_text('{"vendor": "ACM')
        assert not result.success

    def test_plain_text_fails(self):
        result = extract_json_from_text("This is just plain text with no JSON.")
        assert not result.success

    def test_nested_json(self):
        text = '{"line_items": [{"description": "Dev", "amount": 500.0}], "total": 500.0}'
        result = extract_json_from_text(text)
        assert result.success
        assert result.data["line_items"][0]["amount"] == 500.0


# ─── JSON validator tests ─────────────────────────────────────────────────────

class TestJSONValidator:
    def setup_method(self):
        self.validator = JSONValidator()
        self.ref_json = json.dumps({"vendor": "ACME", "total": 100.0, "currency": "USD"})

    def test_valid_prediction(self):
        result = self.validator.validate(self.ref_json, self.ref_json)
        assert result["json_valid"] is True
        assert result["schema_compliant"] is True
        assert result["field_coverage"] == 1.0
        assert result["extra_fields"] == []
        assert result["missing_fields"] == []

    def test_extra_fields_detected(self):
        pred = json.dumps({
            "vendor": "ACME", "total": 100.0, "currency": "USD",
            "confidence_score": 0.98, "model": "gpt-4"
        })
        result = self.validator.validate(pred, self.ref_json)
        assert result["json_valid"] is True
        assert "confidence_score" in result["extra_fields"]
        assert "model" in result["extra_fields"]

    def test_missing_fields_detected(self):
        pred = json.dumps({"vendor": "ACME"})
        result = self.validator.validate(pred, self.ref_json)
        assert result["json_valid"] is True
        assert "total" in result["missing_fields"]

    def test_invalid_json_prediction(self):
        result = self.validator.validate("not json at all", self.ref_json)
        assert result["json_valid"] is False
        assert result["schema_compliant"] is False

    def test_markdown_wrapped_prediction(self):
        pred = f'```json\n{self.ref_json}\n```'
        result = self.validator.validate(pred, self.ref_json)
        assert result["json_valid"] is True


# ─── Metrics computer tests ───────────────────────────────────────────────────

class TestMetricsComputer:
    def setup_method(self):
        self.mc = MetricsComputer()
        self.ref = json.dumps({"vendor": "ACME Corp", "total": 150.0, "currency": "USD"})

    def _make_result(self, prediction: str) -> EvaluationResult:
        return self.mc.compute(
            prediction_text=prediction,
            reference_text=self.ref,
            task_type=TaskType.STRUCTURED_EXTRACTION,
            source_text="Invoice from ACME Corp. Total: $150.00 USD",
        )

    def test_perfect_prediction(self):
        result = self._make_result(self.ref)
        assert result.format_valid is True
        assert result.exact_match is True
        assert result.field_f1 == 1.0

    def test_invalid_json(self):
        result = self._make_result("not json")
        assert result.format_valid is False
        assert result.exact_match is False
        assert result.field_f1 == 0.0
        assert FailureMode.FORMAT_ERROR in result.failure_modes

    def test_truncated_output(self):
        result = self._make_result('{"vendor": "ACME Cor')
        assert result.format_valid is False
        assert FailureMode.TRUNCATED_OUTPUT in result.failure_modes

    def test_verbose_wrapper(self):
        pred = f'Here is the extracted data:\n```json\n{self.ref}\n```\n\nLet me know!'
        result = self._make_result(pred)
        assert result.format_valid is True
        # Instruction not followed — has preamble/postamble
        assert result.instruction_followed is False

    def test_hallucination_detected(self):
        pred_data = json.loads(self.ref)
        pred_data["confidence_score"] = 0.98
        pred = json.dumps(pred_data)
        result = self._make_result(pred)
        assert result.hallucination_detected is True

    def test_alignment_score_range(self):
        result = self._make_result(self.ref)
        assert 0.0 <= result.alignment_score <= 1.0

    def test_alignment_score_better_for_perfect(self):
        perfect = self._make_result(self.ref)
        bad = self._make_result("completely wrong output")
        assert perfect.alignment_score > bad.alignment_score

    def test_failure_mode_none_for_perfect(self):
        result = self._make_result(self.ref)
        assert FailureMode.NONE in result.failure_modes


# ─── Lexical metric tests ─────────────────────────────────────────────────────

class TestLexicalMetrics:
    def test_token_f1_identical(self):
        assert compute_token_f1("hello world", "hello world") == 1.0

    def test_token_f1_no_overlap(self):
        assert compute_token_f1("foo bar", "baz qux") == 0.0

    def test_token_f1_partial(self):
        score = compute_token_f1("the quick brown fox", "the quick fox")
        assert 0.0 < score < 1.0

    def test_token_f1_empty_both(self):
        assert compute_token_f1("", "") == 1.0

    def test_rouge_l_identical(self):
        assert compute_rouge_l("hello world test", "hello world test") == 1.0

    def test_rouge_l_no_overlap(self):
        assert compute_rouge_l("foo bar baz", "qux quux corge") == 0.0

    def test_rouge_l_partial(self):
        score = compute_rouge_l("the cat sat on the mat", "the cat is on the floor")
        assert 0.0 < score < 1.0

    def test_field_f1_perfect(self):
        d = {"name": "ACME", "total": 100.0}
        assert compute_field_f1(d, d) == 1.0

    def test_field_f1_empty_pred(self):
        ref = {"name": "ACME"}
        score = compute_field_f1({}, ref)
        assert score == 0.0

    def test_field_f1_numeric_tolerance(self):
        pred = {"total": 100.0}
        ref = {"total": 100.5}  # Within 2% tolerance
        score = compute_field_f1(pred, ref)
        assert score == 1.0


# ─── Hallucination detector tests ─────────────────────────────────────────────

class TestHallucinationDetector:
    def setup_method(self):
        self.detector = HallucinationDetector()

    def test_no_hallucination(self):
        pred = {"vendor": "ACME", "total": 100.0}
        ref = {"vendor": "ACME", "total": 100.0}
        report = self.detector.analyze(pred, ref, "Invoice from ACME. Total: $100")
        assert not report.hallucination_detected

    def test_extra_fields_hallucination(self):
        pred = {"vendor": "ACME", "total": 100.0, "confidence": 0.99}
        ref = {"vendor": "ACME", "total": 100.0}
        report = self.detector.analyze(pred, ref, "Invoice from ACME")
        assert report.hallucination_detected
        assert "confidence" in report.field_hallucinations

    def test_type_violation(self):
        pred = {"total": "100.0"}  # String instead of float
        ref = {"total": 100.0}
        report = self.detector.analyze(pred, ref, "Total is 100")
        assert report.hallucination_detected
        assert len(report.type_violations) > 0

    def test_quantitative_error(self):
        pred = {"total": 200.0}   # 100% error — way above 2% tolerance
        ref = {"total": 100.0}
        report = self.detector.analyze(pred, ref, "Total: 100")
        assert len(report.quantitative_errors) > 0

    def test_empty_prediction(self):
        report = self.detector.analyze({}, {"vendor": "ACME"}, "")
        assert not report.hallucination_detected


# ─── Tool call metrics tests ──────────────────────────────────────────────────

class TestToolCallMetrics:
    def setup_method(self):
        self.tc = ToolCallMetrics()

    def test_correct_tool_and_args(self):
        pred = {"tool": "process_refund", "args": {"amount": 60.0, "reason": "overcharge"}}
        ref  = {"tool": "process_refund", "args": {"amount": 60.0, "reason": "overcharge"}}
        result = self.tc.evaluate(pred, ref)
        assert result.tool_name_correct is True
        assert result.arg_key_accuracy == 1.0
        assert result.full_match is True

    def test_wrong_tool(self):
        pred = {"tool": "lookup_account", "args": {}}
        ref  = {"tool": "process_refund", "args": {"amount": 60.0}}
        result = self.tc.evaluate(pred, ref)
        assert result.tool_name_correct is False
        assert result.full_match is False

    def test_correct_tool_wrong_args(self):
        pred = {"tool": "process_refund", "args": {"amount": 60.0, "extra_field": "x"}}
        ref  = {"tool": "process_refund", "args": {"amount": 60.0, "reason": "overcharge"}}
        result = self.tc.evaluate(pred, ref)
        assert result.tool_name_correct is True
        assert result.arg_key_accuracy < 1.0

    def test_nested_schema_format(self):
        pred = {"selected_tool": {"tool_name": "search_kb", "arguments": {"query": "test"}}}
        ref  = {"selected_tool": {"tool_name": "search_kb", "arguments": {"query": "test"}}}
        result = self.tc.evaluate(pred, ref)
        assert result.tool_name_correct is True

    def test_none_inputs(self):
        result = self.tc.evaluate(None, None)
        assert not result.tool_name_correct
        assert result.arg_key_accuracy == 0.0


# ─── Latency profiler tests ───────────────────────────────────────────────────

class TestLatencyProfiler:
    def test_start_end_records_measurement(self):
        import time
        profiler = LatencyProfiler()
        start = profiler.start()
        time.sleep(0.01)
        m = profiler.end(start, "req_1", input_tokens=100, output_tokens=50)
        assert m.latency_ms >= 10.0
        assert m.tokens_per_second > 0
        assert len(profiler.measurements) == 1

    def test_summary_with_measurements(self):
        profiler = LatencyProfiler()
        for i in range(10):
            start = profiler.start()
            profiler.end(start, f"req_{i}", input_tokens=100, output_tokens=50)
        summary = profiler.summary()
        assert summary["n_requests"] == 10
        assert "p95_latency_ms" in summary
        assert summary["p95_latency_ms"] >= summary["p50_latency_ms"]

    def test_empty_summary(self):
        profiler = LatencyProfiler()
        summary = profiler.summary()
        assert summary["n_requests"] == 0

    def test_reset(self):
        profiler = LatencyProfiler()
        start = profiler.start()
        profiler.end(start, "r1", input_tokens=10, output_tokens=10)
        profiler.reset()
        assert len(profiler.measurements) == 0


# ─── Score aggregator tests ───────────────────────────────────────────────────

class TestScoreAggregator:
    def _make_result(self, format_valid=True, hallucination=False, field_f1=0.8):
        return EvaluationResult(
            model_stage=ModelStage.SFT,
            task_type=TaskType.STRUCTURED_EXTRACTION,
            difficulty=DifficultyLevel.MEDIUM,
            format_valid=format_valid,
            instruction_followed=format_valid,
            hallucination_detected=hallucination,
            field_f1=field_f1,
            rouge_l=0.7,
            latency_ms=300.0,
        )

    def test_aggregation_structure(self):
        results = [self._make_result() for _ in range(10)]
        agg = ScoreAggregator()
        benchmark = agg.aggregate(results, "test-model", ModelStage.SFT)
        assert benchmark.n_examples == 10
        assert 0.0 <= benchmark.format_valid <= 1.0
        assert 0.0 <= benchmark.hallucination_rate <= 1.0

    def test_format_valid_mean(self):
        results = [
            self._make_result(format_valid=True),
            self._make_result(format_valid=True),
            self._make_result(format_valid=False),
            self._make_result(format_valid=False),
        ]
        agg = ScoreAggregator()
        benchmark = agg.aggregate(results, "test", ModelStage.BASE)
        assert abs(benchmark.format_valid - 0.5) < 0.01

    def test_by_task_breakdown_populated(self):
        results = [self._make_result() for _ in range(5)]
        agg = ScoreAggregator()
        benchmark = agg.aggregate(results, "test", ModelStage.SFT)
        assert "structured_extraction" in benchmark.by_task

    def test_failure_mode_counts(self):
        # pyrefly: ignore [missing-import]
        from src.data.schemas import FailureMode
        r1 = self._make_result(format_valid=False)
        r1.failure_modes = [FailureMode.FORMAT_ERROR]
        r2 = self._make_result(format_valid=False)
        r2.failure_modes = [FailureMode.FORMAT_ERROR]
        r3 = self._make_result()
        r3.failure_modes = [FailureMode.NONE]

        agg = ScoreAggregator()
        benchmark = agg.aggregate([r1, r2, r3], "test", ModelStage.BASE)
        assert benchmark.failure_mode_counts.get("format_error", 0) == 2

    def test_empty_results(self):
        agg = ScoreAggregator()
        benchmark = agg.aggregate([], "test", ModelStage.BASE)
        assert benchmark.n_examples == 0


# ─── Model comparator tests ───────────────────────────────────────────────────

class TestModelComparator:
    def _make_benchmark(self, stage, format_valid=0.7, halluc=0.2):
        # pyrefly: ignore [missing-import]
        from src.data.schemas import BenchmarkResult
        return BenchmarkResult(
            model_id="test-model",
            model_stage=stage,
            n_examples=100,
            format_valid=format_valid,
            instruction_followed=format_valid - 0.05,
            hallucination_rate=halluc,
            avg_alignment_score=format_valid * 0.9,
            avg_latency_ms=350.0,
            p95_latency_ms=600.0,
        )

    def test_add_and_compare(self):
        comparator = ModelComparator()
        comparator.add(self._make_benchmark(ModelStage.BASE, format_valid=0.6))
        comparator.add(self._make_benchmark(ModelStage.SFT, format_valid=0.9))
        assert "base" in comparator.results
        assert "sft" in comparator.results

    def test_delta_vs_baseline(self):
        comparator = ModelComparator()
        comparator.add(self._make_benchmark(ModelStage.BASE, format_valid=0.6))
        comparator.add(self._make_benchmark(ModelStage.SFT, format_valid=0.9))
        deltas = comparator.compute_deltas(baseline_stage="base")
        assert "sft" in deltas
        assert deltas["sft"]["format_valid_delta_pp"] > 0

    def test_print_comparison_table_no_crash(self, capsys):
        comparator = ModelComparator()
        comparator.add(self._make_benchmark(ModelStage.BASE, format_valid=0.6))
        comparator.add(self._make_benchmark(ModelStage.SFT, format_valid=0.9))
        comparator.print_comparison_table()
        captured = capsys.readouterr()
        assert "FORMAT" in captured.out.upper() or "Format" in captured.out
