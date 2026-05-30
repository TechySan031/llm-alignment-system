"""
Preprocessor: converts SFTExample and DPOExample objects to tensors.

The preprocessor is the bridge between the schema layer and the training
layer. It has no knowledge of model architecture or training objectives —
it only knows about tokenisation and chat template formatting.

Input:  list[SFTExample] from DatasetOrchestrator
Output: HuggingFace Dataset with columns: input_ids, attention_mask, labels

Input:  list[SFTExample] for DPO
Output: DatasetDict with columns: prompt, chosen, rejected

The loss masking logic is the single most important implementation here.
See the docstring on format_sft_example for a detailed explanation.
"""
from __future__ import annotations

import json
import logging
import random
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from src.data.schemas import (
    DifficultyLevel,
    DPOExample,
    FailureMode,
    RejectionStrategy,
    Role,
    SFTExample,
    TaskType,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SFT tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def format_sft_example(
    example: SFTExample,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> dict | None:
    """
    Tokenise one SFTExample into training tensors with loss masking.

    Loss masking process:
        1. Format the full conversation (system + user + assistant) with
           the tokenizer's chat template.
        2. Format the prompt-only portion (system + user) with
           add_generation_prompt=True to get the exact prompt boundary.
        3. Tokenise both. The prompt token count is the mask boundary.
        4. labels = [-100] * prompt_len + input_ids[prompt_len:]

    Why this matters for Qwen2.5-7B specifically:
        Qwen2.5-7B-Instruct uses ChatML format. The assistant turn starts
        with the tokens for '<|im_start|>assistant\n'. If the tokenizer's
        apply_chat_template includes those tokens in add_generation_prompt,
        they appear in the prompt boundary and are correctly masked.
        This is model-specific behaviour — always verify with a test print
        of the prompt boundary for any new model family.

    Returns None for examples that exceed max_length after tokenisation.
    These are filtered out rather than truncated to preserve label integrity.
    """
    conv = example.conversation

    # Build message list in HuggingFace messages format
    messages = [
        {"role": t.role.value, "content": t.content}
        for t in conv.turns
    ]
    prompt_messages = [
        {"role": t.role.value, "content": t.content}
        for t in conv.prompt_only
    ]

    # Full sequence (prompt + response)
    try:
        full_text: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text: str = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        logger.warning(
            f"[Preprocessor] Chat template failed for {example.example_id}: {e}"
        )
        return None

    # Tokenise full sequence
    full_enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    # Tokenise prompt to find exact mask boundary
    prompt_enc = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids: list[int] = full_enc["input_ids"]
    attention_mask: list[int] = full_enc["attention_mask"]
    prompt_len: int = len(prompt_enc["input_ids"])

    # Skip examples where the prompt itself exceeds max_length
    if prompt_len >= max_length:
        logger.debug(
            f"[Preprocessor] Prompt alone exceeds max_length "
            f"({prompt_len} >= {max_length}). Skipping {example.example_id}."
        )
        return None

    # Apply loss mask
    labels: list[int] = [-100] * prompt_len + input_ids[prompt_len:]

    if len(labels) != len(input_ids):
        logger.warning(
            f"[Preprocessor] Length mismatch for {example.example_id}: "
            f"labels={len(labels)}, input_ids={len(input_ids)}. Skipping."
        )
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_id": example.example_id,
        "task_type": example.task_type.value,
        "difficulty": example.difficulty.value,
        "source": example.source,
        "weight": example.weight,
    }


def build_sft_datasets(
    sft_examples: list[SFTExample],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    seed: int = 42,
    num_proc: int = 4,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Convert a list of SFTExample objects to HuggingFace Datasets.

    Splits before tokenisation to prevent data leakage.
    Returns (train_ds, val_ds, test_ds) where test_ds contains raw
    SFTExample dicts (not tokenised) for flexible evaluation.

    Args:
        sft_examples:  Output of DatasetOrchestrator.generate_sft_examples()
        tokenizer:     Loaded tokenizer (Qwen2.5 tokenizer in practice)
        max_length:    Hard sequence length limit — examples exceeding this
                       after tokenisation are discarded
        seed:          Shuffle seed
        num_proc:      Tokenisation worker count (CPU cores, not GPU)
        val_fraction:  Fraction of examples for validation
        test_fraction: Fraction of examples for test (kept as raw SFTExample dicts)
    """
    random.seed(seed)
    random.shuffle(sft_examples)

    n = len(sft_examples)
    train_end = int(n * (1.0 - val_fraction - test_fraction))
    val_end = int(n * (1.0 - test_fraction))

    raw_train = sft_examples[:train_end]
    raw_val = sft_examples[train_end:val_end]
    raw_test = sft_examples[val_end:]

    logger.info(
        f"[Preprocessor] Split — train: {len(raw_train):,}, "
        f"val: {len(raw_val):,}, test: {len(raw_test):,}"
    )

    def tokenise_fn(row: dict) -> dict:
        # Reconstruct SFTExample from the serialised dict
        ex = SFTExample.from_dict(row)
        result = format_sft_example(ex, tokenizer, max_length)
        if result is None:
            # Return oversized dummy — filtered out below
            return {
                "input_ids": list(range(max_length + 1)),
                "attention_mask": [1] * (max_length + 1),
                "labels": [-100] * (max_length + 1),
                "example_id": row.get("example_id", ""),
                "task_type": row.get("task_type", ""),
                "difficulty": row.get("difficulty", ""),
                "source": row.get("source", ""),
                "weight": float(row.get("weight", 1.0)),
            }
        return result

    # Serialise SFTExamples to dicts for HuggingFace Dataset
    train_dicts = [ex.to_dict() for ex in raw_train]
    val_dicts = [ex.to_dict() for ex in raw_val]

    train_ds = (
        Dataset.from_list(train_dicts)
        .map(tokenise_fn, num_proc=num_proc, desc="Tokenising train")
        .filter(lambda x: len(x["input_ids"]) <= max_length)
    )
    val_ds = (
        Dataset.from_list(val_dicts)
        .map(tokenise_fn, num_proc=num_proc, desc="Tokenising val")
        .filter(lambda x: len(x["input_ids"]) <= max_length)
    )
    # Test set: raw SFTExample dicts for flexible per-example evaluation
    test_ds = Dataset.from_list([ex.to_dict() for ex in raw_test])

    lengths = [len(x["input_ids"]) for x in train_ds]
    logger.info(
        f"[Preprocessor] Train seq lengths — "
        f"mean: {np.mean(lengths):.0f} | "
        f"p95: {np.percentile(lengths, 95):.0f} | "
        f"max: {max(lengths)}"
    )
    logger.info(
        f"[Preprocessor] After length filter — "
        f"train: {len(train_ds):,}, val: {len(val_ds):,}, test: {len(test_ds):,}"
    )
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# DPO pair construction
# ─────────────────────────────────────────────────────────────────────────────

class RejectionEngine:
    """
    Generates plausible-but-wrong rejected responses for DPO training.

    Six strategies cover the main alignment failure modes:
        HALLUCINATED_FIELDS:  Invents metadata not in the reference
        SCHEMA_VIOLATION:     Correct content, wrong types
        TRUNCATED:            Valid content cut off before completion
        VERBOSE_WRAPPER:      Correct content wrapped in forbidden formatting
        INSTRUCTION_IGNORED:  Ignores an explicit system prompt constraint
        FACTUAL_ERROR:        Substitutes incorrect factual claims

    Strategy selection is weighted by task type — structured output tasks
    get more SCHEMA_VIOLATION and HALLUCINATED_FIELDS, instruction following
    gets more INSTRUCTION_IGNORED and VERBOSE_WRAPPER.
    """

    _WEIGHTS_BY_TASK: dict[TaskType, dict[RejectionStrategy, float]] = {
        TaskType.STRUCTURED_EXTRACTION: {
            RejectionStrategy.HALLUCINATED_FIELDS: 0.30,
            RejectionStrategy.SCHEMA_VIOLATION: 0.25,
            RejectionStrategy.TRUNCATED: 0.20,
            RejectionStrategy.VERBOSE_WRAPPER: 0.15,
            RejectionStrategy.INSTRUCTION_IGNORED: 0.05,
            RejectionStrategy.FACTUAL_ERROR: 0.05,
        },
        TaskType.INSTRUCTION_FOLLOWING: {
            RejectionStrategy.HALLUCINATED_FIELDS: 0.10,
            RejectionStrategy.SCHEMA_VIOLATION: 0.05,
            RejectionStrategy.TRUNCATED: 0.15,
            RejectionStrategy.VERBOSE_WRAPPER: 0.30,
            RejectionStrategy.INSTRUCTION_IGNORED: 0.30,
            RejectionStrategy.FACTUAL_ERROR: 0.10,
        },
        TaskType.TOOL_CALLING: {
            RejectionStrategy.HALLUCINATED_FIELDS: 0.25,
            RejectionStrategy.SCHEMA_VIOLATION: 0.30,
            RejectionStrategy.TRUNCATED: 0.15,
            RejectionStrategy.VERBOSE_WRAPPER: 0.15,
            RejectionStrategy.INSTRUCTION_IGNORED: 0.10,
            RejectionStrategy.FACTUAL_ERROR: 0.05,
        },
        TaskType.ALIGNMENT_EVAL: {
            RejectionStrategy.HALLUCINATED_FIELDS: 0.20,
            RejectionStrategy.SCHEMA_VIOLATION: 0.05,
            RejectionStrategy.TRUNCATED: 0.10,
            RejectionStrategy.VERBOSE_WRAPPER: 0.20,
            RejectionStrategy.INSTRUCTION_IGNORED: 0.25,
            RejectionStrategy.FACTUAL_ERROR: 0.20,
        },
    }

    _DEFAULT_WEIGHTS: dict[RejectionStrategy, float] = {
        RejectionStrategy.HALLUCINATED_FIELDS: 0.20,
        RejectionStrategy.SCHEMA_VIOLATION: 0.15,
        RejectionStrategy.TRUNCATED: 0.20,
        RejectionStrategy.VERBOSE_WRAPPER: 0.25,
        RejectionStrategy.INSTRUCTION_IGNORED: 0.10,
        RejectionStrategy.FACTUAL_ERROR: 0.10,
    }

    def select_strategy(self, task_type: TaskType) -> RejectionStrategy:
        weights = self._WEIGHTS_BY_TASK.get(task_type, self._DEFAULT_WEIGHTS)
        strategies = list(weights.keys())
        probabilities = list(weights.values())
        return random.choices(strategies, weights=probabilities, k=1)[0]

    def generate(
        self,
        chosen: str,
        task_type: TaskType,
        strategy: RejectionStrategy | None = None,
    ) -> tuple[str, RejectionStrategy]:
        """
        Generate a rejected response and return (rejected_text, strategy_used).

        The strategy is returned so it can be stored in DPOExample
        for ablation studies.
        """
        if strategy is None:
            strategy = self.select_strategy(task_type)

        rejected = self._apply_strategy(chosen, strategy)
        return rejected, strategy

    def _apply_strategy(self, chosen: str, strategy: RejectionStrategy) -> str:
        if strategy == RejectionStrategy.HALLUCINATED_FIELDS:
            return self._hallucinate_fields(chosen)
        elif strategy == RejectionStrategy.SCHEMA_VIOLATION:
            return self._violate_schema(chosen)
        elif strategy == RejectionStrategy.TRUNCATED:
            return self._truncate(chosen)
        elif strategy == RejectionStrategy.VERBOSE_WRAPPER:
            return self._verbose_wrap(chosen)
        elif strategy == RejectionStrategy.INSTRUCTION_IGNORED:
            return self._ignore_instruction(chosen)
        elif strategy == RejectionStrategy.FACTUAL_ERROR:
            return self._inject_error(chosen)
        return chosen

    def _hallucinate_fields(self, chosen: str) -> str:
        try:
            obj = json.loads(chosen)
            obj["confidence_score"] = round(random.uniform(0.70, 0.99), 3)
            obj["processed_at"] = "2024-06-15T09:30:00Z"
            obj["extraction_engine"] = "gpt-4-turbo-2024-04-09"
            return json.dumps(obj, indent=2)
        except json.JSONDecodeError:
            return chosen + "\n\nNote: Processed with high confidence (97.3%)."

    def _violate_schema(self, chosen: str) -> str:
        try:
            obj = json.loads(chosen)
            numeric_keys = [k for k, v in obj.items() if isinstance(v, (int, float))]
            bool_keys = [k for k, v in obj.items() if isinstance(v, bool)]
            if numeric_keys:
                k = random.choice(numeric_keys)
                obj[k] = str(obj[k])
            elif bool_keys:
                k = random.choice(bool_keys)
                obj[k] = "yes" if obj[k] else "no"
            return json.dumps(obj, indent=2)
        except json.JSONDecodeError:
            return chosen.upper()

    def _truncate(self, chosen: str) -> str:
        cut = int(len(chosen) * random.uniform(0.40, 0.78))
        return chosen[:cut]

    def _verbose_wrap(self, chosen: str) -> str:
        return (
            "Based on my careful analysis of the provided information, "
            "here is the extracted structured data:\n\n"
            f"```json\n{chosen}\n```\n\n"
            "Please note that I have done my best to extract all available "
            "information. Some fields may require manual verification. "
            "Let me know if you need any clarification."
        )

    def _ignore_instruction(self, chosen: str) -> str:
        try:
            obj = json.loads(chosen)
            return (
                f"I've analysed the text. "
                f"The main points are: {', '.join(str(v) for v in list(obj.values())[:3])}. "
                f"Would you like me to elaborate on any of these?"
            )
        except json.JSONDecodeError:
            return (
                "I've reviewed the content carefully. "
                "There are several important aspects to consider here. "
                f"In summary: {chosen[:100]}..."
            )

    def _inject_error(self, chosen: str) -> str:
        try:
            obj = json.loads(chosen)
            str_keys = [k for k, v in obj.items() if isinstance(v, str) and len(v) > 3]
            if str_keys:
                k = random.choice(str_keys)
                obj[k] = obj[k][::-1][:len(obj[k])]  # Scramble the value
            return json.dumps(obj, indent=2)
        except json.JSONDecodeError:
            return chosen.replace(
                chosen.split()[0] if chosen.split() else "The",
                "According to my analysis,"
            )


_rejection_engine = RejectionEngine()


def build_dpo_example(
    sft_example: SFTExample,
    tokenizer: PreTrainedTokenizer,
) -> DPOExample | None:
    """
    Build one DPOExample from one SFTExample.

    The prompt is the chat-template-formatted prefix up to and including
    the assistant turn start tokens (add_generation_prompt=True).
    TRL's DPOTrainer uses this prefix to locate where chosen/rejected begin
    when computing per-token log-probabilities for the DPO loss.

    Returns None if the SFTExample has no assistant response or if
    the rejection engine produces an identical response to chosen.
    """
    try:
        reference = sft_example.conversation.final_response
        if reference is None:
            return None

        prompt_messages = [
            {"role": t.role.value, "content": t.content}
            for t in sft_example.conversation.prompt_only
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        rejected, strategy = _rejection_engine.generate(
            chosen=reference,
            task_type=sft_example.task_type,
        )

        if rejected.strip() == reference.strip():
            return None

        return DPOExample(
            sft_example_id=sft_example.example_id,
            prompt=prompt,
            chosen=reference,
            rejected=rejected,
            preference_strength=1.0,
            rejection_strategy=strategy,
            task_type=sft_example.task_type,
            difficulty=sft_example.difficulty,
        )
    except Exception as e:
        logger.debug(f"[DPO] Pair build failed for {sft_example.example_id}: {e}")
        return None


def build_dpo_datasets(
    sft_train_examples: list[SFTExample],
    tokenizer: PreTrainedTokenizer,
    seed: int = 42,
) -> DatasetDict:
    """
    Build DPO DatasetDict from the training split of SFTExamples.

    IMPORTANT: only pass the training split here — never test examples.
    The test split is reserved for evaluation and must not contaminate
    the DPO training distribution.

    Returns:
        DatasetDict with keys "train" and "validation".
        Each example has columns: prompt, chosen, rejected
        (the exact format TRL's DPOTrainer expects).
    """
    random.seed(seed)

    pairs: list[dict] = []
    skipped = 0
    for ex in sft_train_examples:
        dpo_ex = build_dpo_example(ex, tokenizer)
        if dpo_ex is not None:
            pairs.append(dpo_ex.to_trl_dict())
        else:
            skipped += 1

    logger.info(
        f"[DPO Preprocessor] Built {len(pairs):,} pairs, "
        f"skipped {skipped} (no response or identical chosen/rejected)"
    )

    random.shuffle(pairs)
    n = len(pairs)
    split = int(n * 0.85)

    logger.info(
        f"[DPO Preprocessor] DPO split — train: {split:,}, val: {n - split:,}"
    )

    return DatasetDict({
        "train": Dataset.from_list(pairs[:split]),
        "validation": Dataset.from_list(pairs[split:]),
    })