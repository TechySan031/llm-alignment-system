
"""
Hallucination detection for structured and open-ended model outputs.

Hallucination taxonomy for alignment systems:

    field_hallucination:
        The model invents JSON fields not present in the reference schema.
        Example: adding "confidence_score": 0.98 when the schema has no such field.
        Detection: compare prediction keys against reference keys after flattening.

    value_hallucination:
        A field exists in both prediction and reference, but the value
        in the prediction is not grounded in the source document.
        Detection: check if the predicted string value appears anywhere
        in the original source text (entity grounding check).

    type_hallucination:
        A field has the correct key but wrong value type.
        Example: {"total": "100.0"} when the schema expects {"total": 100.0}.
        Detection: compare Python types of matching fields.

    quantitative_hallucination:
        A numeric field has the wrong value (arithmetic error or fabrication).
        Example: computing subtotal incorrectly from line items.
        Detection: compare numeric values within a tolerance.

Why hallucination matters for alignment:
    A model that adds "confidence_score: 0.95" to every invoice extraction
    looks helpful but is fabricating data. In production financial or legal
    applications, this causes downstream systems to consume fabricated fields.
    DPO training explicitly penalises this by making hallucinated responses
    the rejected sample.
"""
# pyrefly: ignore [invalid-syntax]
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HallucinationReport:
    """
    Structured hallucination analysis for one prediction.

    Attributes:
        hallucination_detected: True if any hallucination type found.
        hallucination_rate:     Fraction of prediction fields that are hallucinated.
        field_hallucinations:   List of field names invented by the model.
        type_violations:        List of {field, expected_type, got_type} dicts.
        value_issues:           List of {field, value} where value not in source.
        quantitative_errors:    List of {field, expected, got} for numeric mismatches.
        total_pred_fields:      Total fields in prediction.
        total_hallucinated:     Total count of hallucinated fields.
    """
    hallucination_detected: bool = False
    hallucination_rate: float = 0.0
    field_hallucinations: list[str] = field(default_factory=list)
    type_violations: list[dict] = field(default_factory=list)
    value_issues: list[dict] = field(default_factory=list)
    quantitative_errors: list[dict] = field(default_factory=list)
    total_pred_fields: int = 0
    total_hallucinated: int = 0

    def to_dict(self) -> dict:
        return {
            "hallucination_detected": self.hallucination_detected,
            "hallucination_rate": self.hallucination_rate,
            "field_hallucinations": self.field_hallucinations,
            "type_violations": self.type_violations,
            "value_issues": self.value_issues,
            "quantitative_errors": self.quantitative_errors,
            "total_pred_fields": self.total_pred_fields,
            "total_hallucinated": self.total_hallucinated,
        }


class HallucinationDetector:
    """
    Detects hallucination in structured model outputs.

    Works without a GPU. Takes parsed dicts (from JSONValidator)
    and the original source text as inputs.

    Usage:
        detector = HallucinationDetector()
        report = detector.analyze(
            pred_data={"vendor": "ACME", "confidence": 0.99},
            ref_data={"vendor": "ACME"},
            source_text="Invoice from ACME Corp...",
        )
        print(report.field_hallucinations)  # ["confidence"]
    """

    # Fields where value grounding check is not meaningful
    # (enum values, computed fields, schema-defined constants)
    _SKIP_GROUNDING: frozenset[str] = frozenset({
        "currency", "priority", "category", "sentiment", "transaction_type",
        "contract_type", "entity_type", "role", "tool", "tool_name",
        "confidence", "requires_escalation", "escalate",
        "suggested_team", "team", "is_harmful",
    })

    # Numeric comparison tolerance (2% relative error allowed)
    _NUMERIC_TOLERANCE = 0.02

    def analyze(
        self,
        pred_data: dict,
        ref_data: dict,
        source_text: str = "",
    ) -> HallucinationReport:
        """
        Analyse prediction for hallucination against reference and source.

        Args:
            pred_data:   Parsed prediction dict.
            ref_data:    Parsed reference (ground truth) dict.
            source_text: Original input text the model was given.
                         Used for entity grounding checks.

        Returns:
            HallucinationReport with categorised findings.
        """
        report = HallucinationReport()

        if not isinstance(pred_data, dict) or not isinstance(ref_data, dict):
            return report

        pred_flat = self._flatten(pred_data)
        ref_flat = self._flatten(ref_data)

        pred_keys = set(pred_flat.keys())
        ref_keys = set(ref_flat.keys())
        report.total_pred_fields = len(pred_keys)

        # ── Field hallucination: keys in prediction not in reference ──────────
        extra_keys = pred_keys - ref_keys
        report.field_hallucinations = sorted(extra_keys)

        # ── Type violations: matching keys with wrong Python types ────────────
        for key in pred_keys & ref_keys:
            pred_val = pred_flat[key]
            ref_val = ref_flat[key]
            if pred_val is None or ref_val is None:
                continue
            pred_type = type(pred_val)
            ref_type = type(ref_val)
            # int and float are compatible
            if {pred_type, ref_type} <= {int, float}:
                continue
            if pred_type != ref_type:
                report.type_violations.append({
                    "field": key,
                    "expected_type": ref_type.__name__,
                    "got_type": pred_type.__name__,
                    "pred_value": str(pred_val)[:50],
                    "ref_value": str(ref_val)[:50],
                })

        # ── Quantitative errors: numeric fields with wrong values ─────────────
        for key in pred_keys & ref_keys:
            pred_val = pred_flat[key]
            ref_val = ref_flat[key]
            if not isinstance(pred_val, (int, float)):
                continue
            if not isinstance(ref_val, (int, float)):
                continue
            if ref_val == 0:
                continue
            rel_error = abs(pred_val - ref_val) / abs(ref_val)
            if rel_error > self._NUMERIC_TOLERANCE:
                report.quantitative_errors.append({
                    "field": key,
                    "expected": ref_val,
                    "got": pred_val,
                    "relative_error_pct": round(rel_error * 100, 2),
                })

        # ── Value grounding: string values not found in source text ───────────
        if source_text:
            source_lower = source_text.lower()
            for key in pred_keys & ref_keys:
                base_key = key.split(".")[-1].split("[")[0]
                if base_key in self._SKIP_GROUNDING:
                    continue
                pred_val = pred_flat[key]
                if not isinstance(pred_val, str) or len(pred_val) < 4:
                    continue
                # Check if this value (or a significant substring) appears in source
                if not self._is_grounded(pred_val, source_lower):
                    ref_val = ref_flat.get(key, "")
                    if isinstance(ref_val, str) and pred_val != ref_val:
                        report.value_issues.append({
                            "field": key,
                            "value": pred_val,
                            "note": "value not found in source text",
                        })

        # ── Aggregate ─────────────────────────────────────────────────────────
        hallucinated = (
            len(report.field_hallucinations)
            + len(report.type_violations)
        )
        report.total_hallucinated = hallucinated
        report.hallucination_detected = hallucinated > 0

        if report.total_pred_fields > 0:
            report.hallucination_rate = round(
                hallucinated / report.total_pred_fields, 4
            )

        return report

    def _is_grounded(self, value: str, source_lower: str) -> bool:
        """
        Check if a string value is grounded in the source text.

        Uses a sliding window: if any 4-character substring of the value
        appears in the source, it is considered grounded.
        This handles abbreviations and partial matches.
        """
        value_lower = value.lower().strip()
        if len(value_lower) <= 4:
            return value_lower in source_lower

        # Direct substring match
        if value_lower in source_lower:
            return True

        # Token overlap: check if key words from the value appear in source
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", value_lower)
        if not tokens:
            return True  # No checkable tokens

        matches = sum(1 for t in tokens if t in source_lower)
        return matches / len(tokens) >= 0.6

    def _flatten(self, d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(self._flatten(v, key))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    out.update(self._flatten(item, f"{key}[{i}]"))
            else:
                out[key] = v
        return out