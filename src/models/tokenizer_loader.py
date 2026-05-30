"""
Tokenizer loading and chat template formatting.

Centralises all tokenizer-related logic so the rest of the codebase
never calls AutoTokenizer.from_pretrained directly.

Key responsibilities:
    - Load tokenizer with correct padding side for training vs inference
    - Ensure pad_token is always set (critical for batched training)
    - Provide consistent chat template formatting across all pipeline stages
    - Store system prompts as the single source of truth

padding_side matters:
    "right" during SFT training:
        Labels are right-aligned. Padding goes on the right so the loss
        mask (labels=-100) correctly aligns with real response tokens.

    "left" during inference/generation:
        The model generates left-to-right from the prompt. Left-padding
        keeps the prompt content flush-right against the generation boundary.
        Right-padding for generation causes attention patterns to degrade
        because the model attends to padding tokens before the real content.
"""
from __future__ import annotations

import logging
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizer

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompts — single source of truth for training and inference
#
# These must be identical between:
#   - preprocessor.py (SFT training)
#   - preprocessor.py (DPO prompt formatting)
#   - inference/prompts.py (serving)
#   - evaluation/benchmarks.py (evaluation inference)
#
# If they diverge, the model encounters a distribution at inference time
# that differs from training, causing degraded output quality.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS: dict[str, str] = {
    "instruction_following": (
        "You are a helpful, harmless, and honest AI assistant. "
        "Follow the user's instructions precisely. "
        "If the instructions specify a format, length, or style constraint, "
        "adhere to it exactly."
    ),
    "structured_extraction": (
        "You are a precise structured data extraction engine. "
        "Extract the requested information from the provided text and return "
        "a valid JSON object matching the specified schema. "
        "Return ONLY the JSON object — no explanation, no markdown code fences, "
        "no preamble. If a field is not present in the source text, use null."
    ),
    "tool_calling": (
        "You are a function-calling agent. "
        "Given a user request and a catalogue of available tools, "
        "select the single most appropriate tool and construct a valid call. "
        "Return ONLY a JSON object with keys: tool, args, reasoning, confidence. "
        "No additional text."
    ),
    "alignment_eval": (
        "You are an honest, helpful, and harmless AI assistant. "
        "When you are uncertain about a fact, say so explicitly. "
        "Decline requests that could cause harm and offer a constructive alternative. "
        "Follow all constraints specified in the user's message."
    ),
    "summarisation": (
        "You are a precise summarisation assistant. "
        "Summarise the provided text according to the user's instructions. "
        "Maintain factual accuracy — do not add information not present in the source."
    ),
    "question_answering": (
        "You are a knowledgeable assistant. "
        "Answer the user's question accurately and concisely. "
        "If you are not certain of the answer, say so clearly rather than guessing."
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Provide accurate, thoughtful responses to the user's questions."
)


def get_system_prompt(task_type: str) -> str:
    """
    Return the system prompt for a given task type string.

    Args:
        task_type: Task type string matching TaskType enum values.
                   E.g. "instruction_following", "structured_extraction".

    Returns:
        System prompt string. Falls back to DEFAULT_SYSTEM_PROMPT
        if the task type is not in the registry.
    """
    return SYSTEM_PROMPTS.get(task_type, DEFAULT_SYSTEM_PROMPT)


def load_tokenizer(
    model_name: str,
    padding_side: str = "right",
    max_length: int = 2048,
    trust_remote_code: bool = True,
    add_eos_token: bool = True,
) -> PreTrainedTokenizer:
    """
    Load a tokenizer with alignment-system-specific configuration.

    Args:
        model_name:        HuggingFace model name or local path.
        padding_side:      "right" for training, "left" for inference.
        max_length:        Maximum sequence length. Sets model_max_length.
        trust_remote_code: Required for Qwen and other models with custom code.
        add_eos_token:     Ensure EOS token is in the vocabulary.

    Returns:
        Configured PreTrainedTokenizer.

    Common issue — Qwen tokenizer:
        Qwen's tokenizer sets im_end_id as the eos_token in some versions.
        This can cause issues if pad_token falls back to eos_token and they
        are the same as the end-of-turn token. Verify with:
            print(tokenizer.pad_token, tokenizer.pad_token_id)
            print(tokenizer.eos_token, tokenizer.eos_token_id)
    """
    logger.info(
        f"[Tokenizer] Loading: {model_name} | "
        f"padding_side={padding_side} | max_length={max_length}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
        model_max_length=max_length,
    )

    # Ensure pad_token is set — many causal LMs ship without one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(
            f"[Tokenizer] pad_token not set — using eos_token: "
            f"'{tokenizer.pad_token}' (id={tokenizer.pad_token_id})"
        )

    # Verify chat template is available
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        logger.warning(
            "[Tokenizer] No chat template found on this tokenizer. "
            "Chat template formatting is required for SFT and DPO training. "
            "Ensure you are using an instruction-tuned model variant "
            "(e.g. Qwen2.5-7B-Instruct, not Qwen2.5-7B)."
        )

    logger.info(
        f"[Tokenizer] Loaded — "
        f"vocab_size={tokenizer.vocab_size:,} | "
        f"pad_token='{tokenizer.pad_token}' (id={tokenizer.pad_token_id}) | "
        f"eos_token='{tokenizer.eos_token}' (id={tokenizer.eos_token_id}) | "
        f"chat_template={'yes' if tokenizer.chat_template else 'no'}"
    )
    return tokenizer


def format_chat_prompt(
    user_message: str,
    task_type: str,
    tokenizer: PreTrainedTokenizer,
    assistant_prefix: str = "",
    add_generation_prompt: bool = True,
) -> str:
    """
    Format a single-turn prompt using the tokenizer's chat template.

    Used by:
        - Baseline evaluator (inference without fine-tuning)
        - DPO pair builder (prompt formatting)
        - Inference API (serving)

    Args:
        user_message:          The user's input text.
        task_type:             Key for system prompt lookup.
        tokenizer:             Loaded tokenizer with chat template.
        assistant_prefix:      Optional text to prepend to the assistant turn.
                               Used for chain-of-thought prompting.
        add_generation_prompt: Append assistant turn start tokens.
                               Set True for inference, False when the
                               assistant response is included in messages.

    Returns:
        Formatted prompt string ready for tokenisation.
    """
    system = get_system_prompt(task_type)
    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user_message},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as e:
        logger.error(
            f"[Tokenizer] apply_chat_template failed: {e}\n"
            "Falling back to manual ChatML formatting."
        )
        prompt = _manual_chatml_format(messages, add_generation_prompt)

    if assistant_prefix:
        prompt = prompt + assistant_prefix

    return prompt


def format_messages_prompt(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool = True,
) -> str:
    """
    Format a full message list (system + user + optional assistant turns).

    Used for multi-turn conversations and DPO prompt construction.

    Args:
        messages:              List of {"role": ..., "content": ...} dicts.
        tokenizer:             Loaded tokenizer with chat template.
        add_generation_prompt: Append assistant turn start tokens.

    Returns:
        Formatted prompt string.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as e:
        logger.warning(f"[Tokenizer] Template failed: {e}. Using manual format.")
        return _manual_chatml_format(messages, add_generation_prompt)


def _manual_chatml_format(
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> str:
    """
    Manual ChatML formatting fallback for tokenizers without a chat template.

    ChatML format used by Qwen2.5 and many other models:
        <|im_start|>system
        {system content}<|im_end|>
        <|im_start|>user
        {user content}<|im_end|>
        <|im_start|>assistant
        {optional assistant prefix}

    This fallback ensures the pipeline never crashes on missing chat templates,
    but you should always prefer the tokenizer's built-in template.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def count_tokens(
    text: str,
    tokenizer: PreTrainedTokenizer,
) -> int:
    """
    Count the number of tokens in a text string.

    Useful for estimating whether an example will fit within max_length
    before tokenising the full dataset.

    Args:
        text:      Any string.
        tokenizer: Loaded tokenizer.

    Returns:
        Integer token count.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def verify_tokenizer_compatibility(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> dict[str, bool]:
    """
    Run a quick compatibility check on a loaded tokenizer.

    Tests that the tokenizer can handle:
        - Chat template application
        - The expected sequence length
        - Padding and truncation

    Returns:
        Dict of {check_name: passed_bool} for logging.
    """
    checks: dict[str, bool] = {}

    # Chat template check
    test_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": "Hello"},
    ]
    try:
        tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True
        )
        checks["chat_template"] = True
    except Exception:
        checks["chat_template"] = False

    # Pad token check
    checks["pad_token_set"] = tokenizer.pad_token is not None

    # Encoding check
    try:
        ids = tokenizer.encode("test", add_special_tokens=True)
        checks["encoding_works"] = len(ids) > 0
    except Exception:
        checks["encoding_works"] = False

    # Max length check
    checks["max_length_correct"] = tokenizer.model_max_length >= max_length

    failed = [k for k, v in checks.items() if not v]
    if failed:
        logger.warning(f"[Tokenizer] Compatibility checks failed: {failed}")
    else:
        logger.info("[Tokenizer] All compatibility checks passed")

    return checks