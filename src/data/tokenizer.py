"""
Tokenizer utilities.

Centralized tokenizer loading and configuration for:
- SFT training
- DPO training
- Evaluation
- Inference

Supports Qwen, Phi, Llama, Gemma and other HuggingFace models.
"""

from __future__ import annotations

import logging

from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer:
    """
    Load tokenizer from HuggingFace.

    Args:
        model_name:
            Model identifier.

        trust_remote_code:
            Required for some model families such as Qwen.

    Returns:
        Configured tokenizer.
    """

    logger.info(f"Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    # Many causal LMs don't define pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

        logger.info(
            "Pad token missing. Using EOS token as pad token."
        )

    logger.info(
        f"Tokenizer loaded successfully "
        f"(vocab size={tokenizer.vocab_size:,})"
    )

    return tokenizer


def get_chat_template_support(
    tokenizer: PreTrainedTokenizer,
) -> bool:
    """
    Check whether tokenizer supports chat templates.
    """

    return hasattr(tokenizer, "chat_template")


def print_tokenizer_info(
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Print useful tokenizer information.
    """

    logger.info("Tokenizer Information")
    logger.info("---------------------")
    logger.info(f"Vocab Size      : {tokenizer.vocab_size}")
    logger.info(f"Pad Token       : {tokenizer.pad_token}")
    logger.info(f"EOS Token       : {tokenizer.eos_token}")
    logger.info(f"BOS Token       : {tokenizer.bos_token}")
    logger.info(
        f"Chat Template   : "
        f"{get_chat_template_support(tokenizer)}"
    )