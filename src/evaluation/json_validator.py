"""
JSON extraction and schema validation for model outputs.

The model will not always produce clean JSON. Real failure modes observed
in production LLM serving:

    1. Markdown fences:
```json\n{"key": "value"}\n```
       Model added code block formatting despite system prompt saying not to.

    2. Verbose preamble:
           "Here is the extracted data:\n\n{"key": "value"}"
       Model added explanation before the JSON object.

    3. Trailing commentary:
           {"key": "value"}\n\nLet me know if you need anything else.
       Model added text after closing brace.

    4. Single quotes instead of double:
           {'key': 'value'}
       Some models trained on Python code output Python dict syntax.

    5. Truncated output:
           {"key": "val
       Model hit max_new_tokens before completing the JSON.

    6. Nested in markdown:
           Here is the answer:\n\n**JSON Output:**\n```\n{"key": "val"}\n```

Each recovery strategy is tried in order. The first successful parse wins.
If all fail, json_valid=False is recorded and downstream metrics
gracefully return 0.

Schema validation uses the Pydantic models from src/data/schemas.py
but the evaluator is NOT schema-aware by default — it validates the
prediction against the reference structure, not against a hardcoded schema.
This keeps the evaluator generic and reusable for any JSON output task.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


class JSONParseResult:
    """
    Result of attempting to extract and parse JSON from a text string.

    Attributes:
        success:        True if valid JSON was extracted.
        data:           Parsed Python dict/list, or None on failure.
        raw_extracted:  The raw string that was attempted to be parsed.
        strategy:       Which extraction strategy succeeded.
        error:          Error message if parsing failed.
    """
    __slots__ = ("success", "data", "raw_extracted", "strategy", "error")

    def __init__(
        self,
        success: bool,
        data: Optional[dict | list] = None,
        raw_extracted: str = "",
        strategy: str = "none",
        error: str = "",
    ):
        self.success = success
        self.data = data
        self.raw_extracted = raw_extracted
        self.strategy = strategy
        self.error = error

    def __bool__(self) -> bool:
        return self.success


def extract_json_from_text(text: str) -> JSONParseResult:
    """
    Extract and parse a JSON object from model output text.

    Tries six extraction strategies in order of reliability:
        1. direct:         Raw text is valid JSON (ideal case)
        2. stripped:       Strip whitespace then parse
        3. fence_explicit: Extract from ```json ... ``` block
        4. fence_any:      Extract from ``` ... ``` block
        5. first_object:   Find first { ... } span using brace counting
        6. single_to_double: Replace ' with " then parse (Python dict style)

    Returns JSONParseResult with the first successful parse.

    Args:
        text: Raw model output string.

    Returns:
        JSONParseResult — check .success before accessing .data.
    """
    if not text or not text.strip():
        return JSONParseResult(
            success=False, error="Empty model output", strategy="none"
        )

    strategies = [
        ("direct",           _try_direct),
        ("stripped",         _try_stripped),
        ("fence_explicit",   _try_fence_explicit),
        ("fence_any",        _try_fence_any),
        ("first_object",     _try_first_object),
        ("single_to_double", _try_single_to_double),
    ]

    for strategy_name, strategy_fn in strategies:
        result = strategy_fn(text)
        if result is not None:
            logger.debug(f"[JSONValidator] Parsed with strategy: {strategy_name}")
            return JSONParseResult(
                success=True,
                data=result,
                raw_extracted=text,
                strategy=strategy_name,
            )

    return JSONParseResult(
        success=False,
        error=f"All {len(strategies)} extraction strategies failed",
        strategy="all_failed",
        raw_extracted=text,
    )


def _try_direct(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _try_stripped(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _try_fence_explicit(text: str) -> Optional[dict]:
    """Extract from ```json ... ``` blocks."""
    pattern = re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE)
    for match in pattern.finditer(text):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    return None


def _try_fence_any(text: str) -> Optional[dict]:
    """Extract from any ``` ... ``` block."""
    pattern = re.compile(r"```\s*([\s\S]*?)```")
    for match in pattern.finditer(text):
        content = match.group(1).strip()
        if content.startswith("{") or content.startswith("["):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                continue
    return None


def _try_first_object(text: str) -> Optional[dict]:
    """
    Find the first complete JSON object using brace-counting.

    This handles cases where the model wraps JSON in prose:
        "Here is the result: {"key": "val"} Let me know if..."
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _try_single_to_double(text: str) -> Optional[dict]:
    """
    Replace single quotes with double quotes for Python-dict-style output.

    Only applied if the text looks like a Python dict (starts with {').
    This is a last resort — it will mangle strings that contain apostrophes.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    converted = stripped.replace("'", '"')
    try:
        return json.loads(converted)
    except json.JSONDecodeError:
        return None


def validate_against_schema(
    data: dict,
    schema_class: type[BaseModel],
) -> tuple[bool, list[str]]:
    """
    Validate a parsed dict against a Pydantic schema.

    Args:
        data:         Parsed JSON dict from extract_json_from_text.
        schema_class: Pydantic model class to validate against.

    Returns:
        (is_valid, error_messages) tuple.
        is_valid:       True if data passes schema validation.
        error_messages: List of human-readable validation error strings.
                        Empty list when is_valid=True.
    """
    try:
        schema_class.model_validate(data)
        return True, []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            field = " → ".join(str(loc) for loc in err["loc"])
            errors.append(f"{field}: {err['msg']}")
        return False, errors
    except Exception as e:
        return False, [str(e)]


class JSONValidator:
    """
    Stateless validator for model output JSON.

    Combines extraction, parsing, and optional schema validation
    into a single interface used by MetricsComputer.

    Usage:
        validator = JSONValidator()
        result = validator.validate(model_output_text, reference_json_str)
        print(result["json_valid"], result["schema_compliant"])
    """

    def validate(
        self,
        prediction_text: str,
        reference_text: str,
        schema_class: Optional[type[BaseModel]] = None,
    ) -> dict:
        """
        Validate prediction JSON against a reference and optional schema.

        Args:
            prediction_text: Raw model output string.
            reference_text:  Ground truth JSON string.
            schema_class:    Optional Pydantic schema to validate against.

        Returns:
            Dict with keys:
                json_valid:        bool  — prediction is parseable JSON
                schema_compliant:  bool  — prediction matches schema (if provided)
                extraction_strategy: str — which strategy parsed the prediction
                pred_data:         dict | None — parsed prediction
                ref_data:          dict | None — parsed reference
                schema_errors:     list[str] — schema validation errors
                field_coverage:    float — fraction of reference fields present in prediction
                extra_fields:      list[str] — fields in prediction not in reference
                missing_fields:    list[str] — reference fields missing from prediction
        """
        pred_result = extract_json_from_text(prediction_text)
        ref_result = extract_json_from_text(reference_text)

        ref_data = ref_result.data if ref_result.success else {}
        pred_data = pred_result.data if pred_result.success else {}

        result: dict = {
            "json_valid": pred_result.success,
            "schema_compliant": False,
            "extraction_strategy": pred_result.strategy,
            "pred_data": pred_data,
            "ref_data": ref_data,
            "schema_errors": [],
            "field_coverage": 0.0,
            "extra_fields": [],
            "missing_fields": [],
        }

        if not pred_result.success:
            return result

        # Schema validation
        if schema_class is not None and pred_data:
            is_valid, errors = validate_against_schema(pred_data, schema_class)
            result["schema_compliant"] = is_valid
            result["schema_errors"] = errors
        elif pred_data:
            # Without schema: treat as compliant if parseable
            result["schema_compliant"] = True

        # Field coverage analysis
        if isinstance(pred_data, dict) and isinstance(ref_data, dict):
            pred_keys = set(self._flatten(pred_data).keys())
            ref_keys = set(self._flatten(ref_data).keys())

            extra = list(pred_keys - ref_keys)
            missing = list(ref_keys - pred_keys)
            coverage = len(ref_keys - set(missing)) / max(len(ref_keys), 1)

            result["extra_fields"] = extra
            result["missing_fields"] = missing
            result["field_coverage"] = round(coverage, 4)

        return result

    def _flatten(self, d: dict, prefix: str = "") -> dict:
        """Flatten nested dict to dot-notation keys."""
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

