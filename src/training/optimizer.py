"""
Optimizer factory for SFT and DPO training.

AdamW vs Adam — why it matters for transformers:
    Standard Adam incorporates weight decay inside the adaptive update:
        θ_t = θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λθ_{t-1})

    This means weight decay is scaled by the gradient history.
    Parameters with large accumulated gradients get less weight decay
    than parameters with small accumulated gradients. For rare tokens in
    the embedding matrix (large token indices), this means almost no
    regularisation at all.

    AdamW decouples weight decay from the gradient update:
        θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε) - lr · λ · θ_{t-1}

    Weight decay is now proportional only to the parameter value,
    independent of gradient history. All parameters receive the same
    regularisation strength. This is the correct L2 penalty.

Parameter groups — why bias and norm layers need no weight decay:
    Weight decay penalises large parameter values by pulling them toward
    zero. For weight matrices this prevents overfitting. But:

    Bias terms (b in Wx + b):
        A bias close to zero would shift the activation distribution,
        harming learning. Let the optimiser place it where the loss wants.

    LayerNorm scale (γ) and shift (β):
        γ and β directly control normalised activation scale and offset.
        Decaying them towards zero would destroy normalisation.

    Embedding weights:
        Same reasoning as bias — we don't want to systematically push
        all token embeddings toward zero.

paged_adamw_32bit for QLoRA:
    Standard AdamW stores momentum (m) and RMS (v) in fp32:
        40M trainable params × 8 bytes = 320 MB on GPU.

    paged_adamw_32bit (from bitsandbytes) stores optimizer states
    in CPU RAM and pages them to GPU only when needed for parameter
    updates. This moves the 320MB to system RAM, freeing GPU VRAM
    for larger batch sizes or longer sequences.

    Overhead: ~5% slower per step due to CPU↔GPU transfers.
    Worth it: allows training 7B models on 24GB GPUs that would OOM otherwise.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim import AdamW, Optimizer
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)

# These parameter name patterns receive weight_decay=0.0
_NO_DECAY_PATTERNS = frozenset({
    "bias",
    "layer_norm",
    "layernorm",
    "layer_norm_weight",
    "layer_norm_bias",
    "norm",
    "ln_",
    "embedding",
    "embed_tokens",
})


def _is_no_decay(param_name: str) -> bool:
    """Return True if the parameter should NOT receive weight decay."""
    name_lower = param_name.lower()
    return any(pattern in name_lower for pattern in _NO_DECAY_PATTERNS)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    use_paged: bool = True,
) -> Optimizer:
    """
    Build AdamW optimizer with separate parameter groups.

    Args:
        model:         Model with LoRA adapters injected.
        learning_rate: Peak learning rate (reached after warmup).
        weight_decay:  L2 regularisation strength for weight matrices.
        beta1:         Adam first moment decay (momentum). Default 0.9.
        beta2:         Adam second moment decay (RMS). Default 0.999.
        epsilon:       Numerical stability constant. Default 1e-8.
        use_paged:     Use paged_adamw_32bit (requires bitsandbytes + CUDA).
                       Automatically disabled on CPU.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_no_decay(name):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
            decay_names.append(name)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    decay_count = sum(p.numel() for p in decay_params)
    no_decay_count = sum(p.numel() for p in no_decay_params)

    logger.info(
        f"[Optimizer] Parameter groups:\n"
        f"  With weight decay:    {len(decay_params)} params, "
        f"{decay_count / 1e6:.2f}M total\n"
        f"  Without weight decay: {len(no_decay_params)} params, "
        f"{no_decay_count / 1e6:.2f}M total\n"
        f"  lr={learning_rate:.2e} | wd={weight_decay} | "
        f"β=({beta1},{beta2}) | ε={epsilon}"
    )

    # Use paged AdamW on CUDA, standard AdamW on CPU
    if use_paged and torch.cuda.is_available():
        try:
            # pyrefly: ignore [missing-import]
            from bitsandbytes.optim import PagedAdamW32bit
            optimizer = PagedAdamW32bit(
                param_groups,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=epsilon,
            )
            logger.info("[Optimizer] Using PagedAdamW32bit (optimizer states in CPU RAM)")
            return optimizer
        except ImportError:
            logger.warning(
                "[Optimizer] bitsandbytes not available. "
                "Falling back to standard AdamW."
            )

    # Standard AdamW with fused kernel (faster on GPU, works on CPU too)
    try:
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            fused=torch.cuda.is_available(),  # Fused only on CUDA
        )
        fused_str = "fused" if torch.cuda.is_available() else "standard"
        logger.info(f"[Optimizer] Using AdamW ({fused_str})")
    except TypeError:
        # PyTorch < 2.0 does not have fused parameter
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
        )
        logger.info("[Optimizer] Using AdamW (legacy, no fused kernel)")

    return optimizer


def get_optimizer_stats(optimizer: Optimizer) -> dict:
    """
    Return current optimizer state for logging.

    Reads the actual learning rate from the optimizer state dict,
    which reflects the current LR after scheduler updates.
    """
    stats: dict = {}
    for i, group in enumerate(optimizer.param_groups):
        stats[f"group_{i}_lr"] = group["lr"]
        stats[f"group_{i}_wd"] = group["weight_decay"]
        stats[f"group_{i}_n_params"] = len(group["params"])
    return stats