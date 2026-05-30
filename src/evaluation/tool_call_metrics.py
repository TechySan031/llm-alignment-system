"""
Metrics for tool/function calling evaluation.

Tool calling is a binary task in two parts:
    1. Tool selection:  Did the model pick the correct tool name?
    2. Argument construction: Are the arguments correct and complete?

These are evaluated separately because a model can select the right tool
but construct wrong arguments, or vice versa. Both failure modes have
different implications:
    Wrong tool:       Fundamental misunderstanding of user intent
    Wrong arguments:  Understanding intent but failing schema details

Argument accuracy uses Jaccard similarity on argument key sets:
    |pred_args ∩ ref_args| / |pred_args ∪ ref_args|

Value accuracy additionally checks whether argument values match:
    For each matching key: are the values equivalent?
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCallResult:
    """Result of evaluating one tool call prediction."""
    tool_name_correct: bool = False
    predicted_tool: str = ""
    reference_tool: str = ""
    arg_key_accuracy: float = 0.0    # Jaccard on argument keys
    arg_value_accuracy: float = 0.0  # Fraction of matching keys with correct values
    full_match: bool = False          # Tool name correct AND all args match

    def to_dict(self) -> dict:
        return {
            "tool_name_correct": self.tool_name_correct,
            "predicted_tool": self.predicted_tool,
            "reference_tool": self.reference_tool,
            "arg_key_accuracy": self.arg_key_accuracy,
            "arg_value_accuracy": self.arg_value_accuracy,
            "full_match": self.full_match,
        }


class ToolCallMetrics:
    """
    Evaluates tool call predictions against ground truth.

    Extracts tool name and arguments from both prediction and reference
    and computes granular accuracy metrics.

    Expected JSON structure (matches ToolCallResponse schema):
        {
            "tool": "process_refund",
            "args": {"amount": 60.0, "reason": "overcharge"},
            "reasoning": "...",
            "confidence": 0.95
        }

    Alternatively handles TRL's DPO format where the selected_tool
    is nested under "selected_tool.tool_name".
    """

    def evaluate(
        self,
        pred_data: Optional[dict],
        ref_data: Optional[dict],
    ) -> ToolCallResult:
        """
        Evaluate a tool call prediction against the reference.

        Args:
            pred_data: Parsed prediction dict. None if JSON parsing failed.
            ref_data:  Parsed reference dict.

        Returns:
            ToolCallResult with granular accuracy metrics.
        """
        result = ToolCallResult()

        if not isinstance(pred_data, dict) or not isinstance(ref_data, dict):
            return result

        # Extract tool name from both (handle multiple schema formats)
        pred_tool = self._extract_tool_name(pred_data)
        ref_tool = self._extract_tool_name(ref_data)

        result.predicted_tool = pred_tool
        result.reference_tool = ref_tool
        result.tool_name_correct = (
            pred_tool.strip().lower() == ref_tool.strip().lower()
            and pred_tool != ""
        )

        # Extract arguments from both
        pred_args = self._extract_args(pred_data)
        ref_args = self._extract_args(ref_data)

        result.arg_key_accuracy = self._jaccard_keys(pred_args, ref_args)
        result.arg_value_accuracy = self._value_match_rate(pred_args, ref_args)
        result.full_match = (
            result.tool_name_correct
            and result.arg_key_accuracy >= 0.8
            and result.arg_value_accuracy >= 0.8
        )

        return result

    def _extract_tool_name(self, data: dict) -> str:
        """Extract tool name from various schema formats."""
        # Format 1: {"tool": "tool_name", ...}
        if "tool" in data:
            return str(data["tool"])

        # Format 2: {"selected_tool": {"tool_name": "...", ...}}
        selected = data.get("selected_tool", {})
        if isinstance(selected, dict):
            if "tool_name" in selected:
                return str(selected["tool_name"])
            if "name" in selected:
                return str(selected["name"])

        # Format 3: {"function": {"name": "..."}}
        function = data.get("function", {})
        if isinstance(function, dict) and "name" in function:
            return str(function["name"])

        return ""

    def _extract_args(self, data: dict) -> dict:
        """Extract arguments dict from various schema formats."""
        # Format 1: {"args": {...}}
        if "args" in data and isinstance(data["args"], dict):
            return data["args"]

        # Format 2: {"selected_tool": {"arguments": {...}}}
        selected = data.get("selected_tool", {})
        if isinstance(selected, dict):
            if "arguments" in selected and isinstance(selected["arguments"], dict):
                return selected["arguments"]

        # Format 3: {"parameters": {...}}
        if "parameters" in data and isinstance(data["parameters"], dict):
            return data["parameters"]

        # Format 4: {"function": {"arguments": "{...}"}} (OpenAI format)
        function = data.get("function", {})
        if isinstance(function, dict) and "arguments" in function:
            import json
            try:
                return json.loads(function["arguments"])
            except Exception:
                pass

        return {}

    def _jaccard_keys(self, pred_args: dict, ref_args: dict) -> float:
        """Jaccard similarity on argument key sets."""
        if not ref_args and not pred_args:
            return 1.0
        if not ref_args or not pred_args:
            return 0.0
        pred_keys = set(pred_args.keys())
        ref_keys = set(ref_args.keys())
        intersection = len(pred_keys & ref_keys)
        union = len(pred_keys | ref_keys)
        return round(intersection / union, 4) if union > 0 else 0.0

    def _value_match_rate(self, pred_args: dict, ref_args: dict) -> float:
        """
        Fraction of matching argument keys where values also match.

        String values: case-insensitive substring match
        Numeric values: within 2% relative tolerance
        Boolean values: exact match
        """
        common_keys = set(pred_args.keys()) & set(ref_args.keys())
        if not common_keys:
            return 0.0

        matches = 0
        for key in common_keys:
            pred_val = pred_args[key]
            ref_val = ref_args[key]
            if self._values_match(pred_val, ref_val):
                matches += 1

        return round(matches / len(common_keys), 4)

    def _values_match(self, pred: Any, ref: Any) -> bool:
        """Check if two argument values are equivalent."""
        if pred is None and ref is None:
            return True
        if pred is None or ref is None:
            return False
        if isinstance(ref, bool) and isinstance(pred, bool):
            return pred == ref
        if isinstance(ref, (int, float)) and isinstance(pred, (int, float)):
            if ref == 0:
                return pred == 0
            return abs(pred - ref) / abs(ref) <= 0.02
        if isinstance(ref, str) and isinstance(pred, str):
            return (
                pred.lower().strip() == ref.lower().strip()
                or pred.lower().strip() in ref.lower().strip()
                or ref.lower().strip() in pred.lower().strip()
            )
        return str(pred).lower() == str(ref).lower()
