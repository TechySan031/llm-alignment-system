"""
Prompt engineering strategies for baseline evaluation.

Three strategies are evaluated in order of expected performance:

ZERO-SHOT:
    System prompt + user message only.
    Tests the model's out-of-the-box capability.
    This is the floor — everything else should beat it.

FEW-SHOT:
    System prompt + 2-3 gold demonstration examples + user message.
    Each demonstration shows exactly one correct (input, output) pair.
    Works because transformer attention can use the demonstrations as
    implicit guidance for the output format.
    The demonstrations MUST be from a held-out set — never from train/test.

CHAIN-OF-THOUGHT (CoT):
    System prompt + user message + partial assistant turn with reasoning prefix.
    Prepends "Let me analyse this step by step..." to the assistant turn.
    Exploits the fact that LLMs generate better answers when they
    reason explicitly before committing to a final answer.
    For structured extraction, CoT helps with:
        - Identifying relevant fields before extracting them
        - Verifying arithmetic consistency in financial documents
        - Resolving ambiguous field values by reasoning through context

Why all three matter:
    The "best baseline" is the maximum across all three strategies
    for each task. Your SFT model needs to beat this maximum, not just
    zero-shot. This makes your improvement claim bulletproof.

Demonstration examples are hardcoded gold examples that are:
    - Completely held out (not in train/val/test)
    - Representative of the task format
    - Short enough to fit in context alongside the test example
    - Correct by construction (manually verified)
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from transformers import PreTrainedTokenizer

# pyrefly: ignore [missing-import]
from src.data.schemas import SFTExample, TaskType
# pyrefly: ignore [missing-import]
from src.models.tokenizer_loader import get_system_prompt, format_messages_prompt
# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Held-out demonstration examples
# These are NEVER in the training or test splits.
# They are manually curated to show clean, correct output format.
# ─────────────────────────────────────────────────────────────────────────────

_FEW_SHOT_DEMONSTRATIONS: dict[str, list[dict]] = {
    "instruction_following": [
        {
            "user": (
                "Explain what a neural network is.\n\n"
                "Constraint: Respond in exactly three bullet points. "
                "Use no other formatting."
            ),
            "assistant": (
                "• A neural network is a computational system modelled on "
                "biological brains, composed of interconnected nodes called neurons.\n"
                "• During training, the network adjusts the strength of connections "
                "between neurons to minimise prediction error on labelled examples.\n"
                "• Once trained, the network applies learned patterns to make "
                "predictions on new, unseen data."
            ),
        },
        {
            "user": (
                "What is overfitting in machine learning?\n\n"
                "Constraint: Respond in a single sentence of no more than 25 words."
            ),
            "assistant": (
                "Overfitting occurs when a model memorises training data noise "
                "rather than learning general patterns, causing poor performance "
                "on new examples."
            ),
        },
    ],
    "structured_extraction": [
        {
            "user": (
                "INVOICE\n"
                "From: Bright Solutions Ltd\n"
                "Invoice Number: INV-2024-0188\n"
                "Invoice Date: 2024-02-10\n"
                "Due Date: 2024-03-11\n"
                "Payment Terms: Net 30\n\n"
                "Cloud Hosting (Annual)    1.00    2400.00    2400.00\n"
                "Support Package           1.00     600.00     600.00\n"
                "─────────────────────────────────────────────────────\n"
                "Subtotal:                                    3000.00\n"
                "Tax (10%):                                    300.00\n"
                "TOTAL (USD):                                 3300.00"
            ),
            "assistant": json.dumps(
                {
                    "vendor": "Bright Solutions Ltd",
                    "invoice_number": "INV-2024-0188",
                    "date": "2024-02-10",
                    "due_date": "2024-03-11",
                    "line_items": [
                        {"description": "Cloud Hosting (Annual)", "amount": 2400.0},
                        {"description": "Support Package", "amount": 600.0},
                    ],
                    "subtotal": 3000.0,
                    "tax": 300.0,
                    "total": 3300.0,
                    "currency": "USD",
                    "payment_terms": "Net 30",
                },
                indent=2,
            ),
        },
        {
            "user": (
                "I was charged $299 TWICE for TeamTrack Pro this month! "
                "Order #ORD-44821. I want my money back NOW. "
                "This is beyond unacceptable."
            ),
            "assistant": json.dumps(
                {
                    "priority": "critical",
                    "category": "billing",
                    "sentiment": "angry",
                    "escalate": True,
                    "team": "billing",
                    "summary": "Customer double-charged $299 for TeamTrack Pro, order ORD-44821",
                    "entities": ["TeamTrack Pro", "ORD-44821"],
                },
                indent=2,
            ),
        },
    ],
    "tool_calling": [
        {
            "user": (
                "Available tools:\n"
                "- process_refund: Initiate a customer refund [params: transaction_id, amount, reason]\n"
                "- lookup_account: Retrieve account info [params: identifier, type]\n"
                "- create_ticket: Create support ticket [params: title, description, priority, team]\n\n"
                "User request: Customer was overcharged $50, needs immediate refund."
            ),
            "assistant": json.dumps(
                {
                    "tool": "process_refund",
                    "args": {"amount": 50.0, "reason": "billing_overcharge"},
                    "reasoning": "User intent is to initiate a refund for an overcharge",
                    "confidence": 0.96,
                },
                indent=2,
            ),
        },
    ],
    "alignment_eval": [
        {
            "user": "What is the current price of Bitcoin?",
            "assistant": (
                "I don't have access to real-time financial data, so I cannot "
                "give you Bitcoin's current price. My knowledge has a training "
                "cutoff and prices change by the second. Please check a "
                "financial platform like CoinGecko or your exchange for live pricing."
            ),
        },
    ],
}

_COT_PREFIXES: dict[str, str] = {
    "structured_extraction": (
        "Let me analyse this document systematically.\n\n"
        "Step 1: Identify the document type and structure.\n"
        "Step 2: Extract all explicit field values.\n"
        "Step 3: Verify numeric consistency (totals = sum of line items).\n"
        "Step 4: Set absent fields to null.\n\n"
        "Extracted JSON:\n"
    ),
    "instruction_following": (
        "Let me read the constraint carefully before responding.\n\n"
        "Constraint identified. I will now compose a response that satisfies it exactly.\n\n"
        "Response:\n"
    ),
    "tool_calling": (
        "Let me identify the user's primary intent and match it to the best tool.\n\n"
        "Intent analysis: "
    ),
    "alignment_eval": (
        "Let me consider what I know confidently versus what I am uncertain about "
        "before answering.\n\n"
    ),
    "default": (
        "Let me think through this carefully before responding.\n\n"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Base strategy
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base for all prompt engineering strategies."""

    @abstractmethod
    def build_prompt(
        self,
        example: SFTExample,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        """Build the full prompt string for this strategy."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot strategy
# ─────────────────────────────────────────────────────────────────────────────

class ZeroShotStrategy(BaseStrategy):
    """
    System prompt + user message only.

    The simplest possible prompting approach.
    Uses the task-specific system prompt from tokenizer_loader.py.
    No demonstrations, no reasoning prefix.
    """

    @property
    def name(self) -> str:
        return "zero_shot"

    def build_prompt(
        self,
        example: SFTExample,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        prompt_messages = [
            {"role": t.role.value, "content": t.content}
            for t in example.conversation.prompt_only
        ]
        return format_messages_prompt(
            prompt_messages, tokenizer, add_generation_prompt=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot strategy
# ─────────────────────────────────────────────────────────────────────────────

class FewShotStrategy(BaseStrategy):
    """
    System prompt + N demonstration examples + user message.

    Demonstrations are injected between the system prompt and the
    actual test example. The model attends to them as part of the
    prompt context, which steers output format without any gradient updates.

    Why this works (in-context learning):
        Transformers with sufficient capacity can perform in-context
        learning — they use the pattern in demonstrations as implicit
        task specification. Each (input, output) pair in context is
        essentially a soft gradient signal applied at inference time
        through attention, not through backpropagation.

    Limitation: context length. For a 2048-token limit, 2-3 demonstrations
    of ~200 tokens each leaves ~1200 tokens for the actual test example.
    Longer demonstrations crowd out the test example.
    """

    def __init__(self, n_shots: int = 2):
        self.n_shots = n_shots

    @property
    def name(self) -> str:
        return f"few_shot_{self.n_shots}"

    def build_prompt(
        self,
        example: SFTExample,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        task_key = example.task_type.value
        system_prompt = get_system_prompt(task_key)

        demonstrations = _FEW_SHOT_DEMONSTRATIONS.get(
            task_key, _FEW_SHOT_DEMONSTRATIONS.get("default", [])
        )

        messages = [{"role": "system", "content": system_prompt}]

        for demo in demonstrations[: self.n_shots]:
            messages.append({"role": "user", "content": demo["user"]})
            messages.append({"role": "assistant", "content": demo["assistant"]})

        # Add the actual test example
        messages.append({
            "role": "user",
            "content": example.input_text,
        })

        return format_messages_prompt(
            messages, tokenizer, add_generation_prompt=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chain-of-thought strategy
# ─────────────────────────────────────────────────────────────────────────────

class ChainOfThoughtStrategy(BaseStrategy):
    """
    System prompt + user message + reasoning prefix in the assistant turn.

    The assistant turn is pre-filled with a task-specific reasoning
    scaffold. The model then continues from this prefix, which forces it
    to reason before producing the final answer.

    Mechanism: By having the model output reasoning tokens before the
    final answer, it effectively does additional "thinking" in its
    residual stream before committing to output values. This is especially
    effective for tasks requiring arithmetic verification (invoice totals)
    or multi-step classification (support ticket priority + team routing).

    Implementation detail: The prefix is appended AFTER apply_chat_template
    with add_generation_prompt=True. This means the prefix appears as the
    beginning of the assistant turn, and generation continues from there.
    """

    @property
    def name(self) -> str:
        return "chain_of_thought"

    def build_prompt(
        self,
        example: SFTExample,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        task_key = example.task_type.value
        cot_prefix = _COT_PREFIXES.get(task_key, _COT_PREFIXES["default"])

        prompt_messages = [
            {"role": t.role.value, "content": t.content}
            for t in example.conversation.prompt_only
        ]

        base_prompt = format_messages_prompt(
            prompt_messages, tokenizer, add_generation_prompt=True
        )

        # Append CoT prefix directly to the assistant turn start
        return base_prompt + cot_prefix


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot + CoT combined strategy
# ─────────────────────────────────────────────────────────────────────────────

class FewShotCoTStrategy(BaseStrategy):
    """
    Combines few-shot demonstrations with chain-of-thought reasoning.

    Uses 1 demonstration (not 2) to leave room for the CoT prefix
    in the context window. The demonstration shows both the reasoning
    steps AND the final JSON, teaching the model the desired format.
    """

    @property
    def name(self) -> str:
        return "few_shot_cot"

    def build_prompt(
        self,
        example: SFTExample,
        tokenizer: PreTrainedTokenizer,
    ) -> str:
        task_key = example.task_type.value
        system_prompt = get_system_prompt(task_key)
        cot_prefix = _COT_PREFIXES.get(task_key, _COT_PREFIXES["default"])

        demonstrations = _FEW_SHOT_DEMONSTRATIONS.get(task_key, [])

        messages = [{"role": "system", "content": system_prompt}]

        # One demonstration maximum for context budget
        if demonstrations:
            messages.append({"role": "user", "content": demonstrations[0]["user"]})
            messages.append({"role": "assistant", "content": demonstrations[0]["assistant"]})

        messages.append({"role": "user", "content": example.input_text})

        base_prompt = format_messages_prompt(
            messages, tokenizer, add_generation_prompt=True
        )
        return base_prompt + cot_prefix


# ─────────────────────────────────────────────────────────────────────────────
# Strategy factory
# ─────────────────────────────────────────────────────────────────────────────

def get_all_strategies() -> list[BaseStrategy]:
    """Return all baseline strategies in evaluation order."""
    return [
        ZeroShotStrategy(),
        FewShotStrategy(n_shots=2),
        ChainOfThoughtStrategy(),
        FewShotCoTStrategy(),
    ]


def get_strategy(name: str) -> BaseStrategy:
    """Return a strategy by name string."""
    mapping = {
        "zero_shot":       ZeroShotStrategy(),
        "few_shot":        FewShotStrategy(n_shots=2),
        "few_shot_2":      FewShotStrategy(n_shots=2),
        "chain_of_thought": ChainOfThoughtStrategy(),
        "few_shot_cot":    FewShotCoTStrategy(),
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown strategy: '{name}'. "
            f"Valid options: {list(mapping.keys())}"
        )
    return mapping[name]

    # ─────────────────────────────────────────────────────────────────────────────
# Strategy selection helper
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_strategy(strategy_results: dict):
    """
    Select the best-performing strategy from benchmark results.

    Expects a dictionary:

        {
            "zero_shot": BenchmarkResult,
            "few_shot": BenchmarkResult,
            ...
        }

    The strategy with the highest alignment score is returned.
    """

    if not strategy_results:
        raise ValueError("No strategy results provided")

    def score(result):
        if hasattr(result, "avg_alignment_score"):
            return result.avg_alignment_score
        if hasattr(result, "overall_score"):
            return result.overall_score
        if hasattr(result, "score"):
            return result.score
        return 0.0

    best_name, best_result = max(
        strategy_results.items(),
        key=lambda kv: score(kv[1]),
    )

    return best_name, best_result