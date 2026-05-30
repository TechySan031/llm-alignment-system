"""
Perplexity computation for language model evaluation.

Perplexity = exp(H) where H is the cross-entropy of the model
on a held-out text corpus.

    PPL = exp(-1/N * Σ log P(token_i | token_{1..i-1}))

Lower perplexity = model assigns higher probability to the text.

Why perplexity for alignment evaluation:
    After DPO training, perplexity on a general language corpus should
    stay approximately equal to the SFT model's perplexity.
    A large perplexity increase indicates catastrophic forgetting —
    the DPO training shifted the distribution too far from the original.

Sliding window for long texts:
    For texts longer than max_length tokens, we use a sliding window
    to compute perplexity without truncating. Only new tokens in each
    window step contribute to the NLL sum, avoiding double-counting.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


@torch.inference_mode()
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    max_length: int = 1024,
    stride: int = 512,
    batch_size: int = 1,
    show_progress: bool = True,
) -> dict:
    """
    Compute perplexity on a list of texts using a sliding window.

    Args:
        model:         Loaded language model. Must have use_cache=True.
        tokenizer:     Matching tokenizer.
        texts:         List of text strings to evaluate perplexity on.
        max_length:    Window size in tokens.
        stride:        How many new tokens to process per window step.
                       stride=max_length: no overlap (faster, less accurate)
                       stride=max_length/2: 50% overlap (slower, more accurate)
        batch_size:    Texts to process simultaneously.
                       Use 1 for CPU, 4-8 for GPU.
        show_progress: Show tqdm progress bar.

    Returns:
        Dict with:
            perplexity:        Mean perplexity across all texts.
            mean_nll:          Mean negative log-likelihood.
            per_text_ppl:      Per-text perplexity values.
            n_texts:           Number of texts evaluated.
            n_tokens:          Total tokens processed.
    """
    model.eval()
    device = next(model.parameters()).device
    all_nlls: list[float] = []
    per_text_ppl: list[float] = []
    total_tokens = 0

    iterator = tqdm(texts, desc="Computing perplexity") if show_progress else texts

    for text in iterator:
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings["input_ids"][0]
        seq_len = input_ids.size(0)
        text_nlls: list[float] = []
        text_tokens = 0

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end

            # Slice the window
            chunk_ids = input_ids[begin:end].unsqueeze(0).to(device)

            # Labels: -100 for context tokens (already seen), real labels for new tokens
            labels = chunk_ids.clone()
            labels[0, :-target_len] = -100

            try:
                if device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(chunk_ids, labels=labels)
                else:
                    outputs = model(chunk_ids, labels=labels)

                nll = outputs.loss.item()
                if not math.isnan(nll) and not math.isinf(nll):
                    text_nlls.append(nll * target_len)
                    text_tokens += target_len
                    total_tokens += target_len

            except Exception as e:
                logger.warning(f"Perplexity computation failed on chunk: {e}")

            prev_end = end
            if end == seq_len:
                break

        if text_nlls and text_tokens > 0:
            mean_nll = sum(text_nlls) / text_tokens
            ppl = math.exp(mean_nll)
            per_text_ppl.append(ppl)
            all_nlls.extend(text_nlls)

    if not per_text_ppl:
        return {
            "perplexity": float("inf"),
            "mean_nll": float("inf"),
            "per_text_ppl": [],
            "n_texts": 0,
            "n_tokens": 0,
        }

    mean_ppl = float(np.mean(per_text_ppl))
    mean_nll = sum(all_nlls) / max(total_tokens, 1)

    logger.info(
        f"[Perplexity] n={len(per_text_ppl)} texts | "
        f"mean_ppl={mean_ppl:.2f} | "
        f"tokens={total_tokens:,}"
    )

    return {
        "perplexity": round(mean_ppl, 3),
        "mean_nll": round(mean_nll, 6),
        "per_text_ppl": [round(p, 3) for p in per_text_ppl],
        "n_texts": len(per_text_ppl),
        "n_tokens": total_tokens,
    }