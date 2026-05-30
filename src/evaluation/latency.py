"""
Latency profiling for model inference.

Tracks per-request timing and computes statistical summaries.
Latency is measured as wall-clock time from when the tokenised input
is ready to when the generated token IDs are decoded back to text.

Why latency matters in evaluation:
    DPO alignment sometimes increases latency because the aligned model
    produces longer, more careful responses. Tracking latency across
    Base → SFT → DPO shows this trade-off quantitatively.

    For production deployment, p95 latency (not mean) is the relevant SLA metric.
    A model with mean=300ms but p95=2000ms will cause timeouts for 5% of users.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyMeasurement:
    """Single request latency measurement."""
    request_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_stage: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return round(self.total_tokens / (self.latency_ms / 1000), 1)

    @property
    def ms_per_output_token(self) -> float:
        if self.output_tokens <= 0 or self.latency_ms <= 0:
            return 0.0
        return round(self.latency_ms / self.output_tokens, 2)


class LatencyProfiler:
    """
    Collects and summarises latency measurements across an evaluation run.

    Usage:
        profiler = LatencyProfiler()

        start = profiler.start()
        output = model.generate(inputs)
        measurement = profiler.end(
            start, request_id="ex_001",
            input_tokens=100, output_tokens=50
        )
        summary = profiler.summary()
    """

    def __init__(self):
        self.measurements: list[LatencyMeasurement] = []

    def start(self) -> float:
        """Record the start time of an inference call."""
        return time.perf_counter()

    def end(
        self,
        start_time: float,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        model_stage: str = "unknown",
    ) -> LatencyMeasurement:
        """
        Record the end of an inference call and store the measurement.

        Args:
            start_time:   Value returned by self.start().
            request_id:   Unique identifier for this request.
            input_tokens: Number of prompt tokens.
            output_tokens: Number of generated tokens.
            model_stage:  Which model (base/sft/dpo) made this call.

        Returns:
            LatencyMeasurement for this request.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        measurement = LatencyMeasurement(
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(elapsed_ms, 2),
            model_stage=model_stage,
        )
        self.measurements.append(measurement)
        return measurement

    def summary(self) -> dict:
        """
        Compute statistical summary across all measurements.

        Returns:
            Dict with p50, p95, p99, mean latency, mean TPS,
            total requests, total tokens.
        """
        if not self.measurements:
            return {
                "n_requests": 0,
                "mean_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "mean_tokens_per_second": 0.0,
                "mean_ms_per_output_token": 0.0,
                "total_tokens": 0,
            }

        latencies = [m.latency_ms for m in self.measurements]
        tps = [m.tokens_per_second for m in self.measurements]
        ms_per_tok = [m.ms_per_output_token for m in self.measurements if m.output_tokens > 0]

        return {
            "n_requests": len(self.measurements),
            "mean_latency_ms": round(float(np.mean(latencies)), 1),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 1),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 1),
            "mean_tokens_per_second": round(float(np.mean(tps)), 1),
            "mean_ms_per_output_token": round(float(np.mean(ms_per_tok)), 2) if ms_per_tok else 0.0,
            "total_tokens": sum(m.total_tokens for m in self.measurements),
        }

    def reset(self) -> None:
        """Clear all stored measurements."""
        self.measurements.clear()
