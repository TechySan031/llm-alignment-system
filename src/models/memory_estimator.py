"""
VRAM and system memory estimation for model loading and training.

Used to:
    1. Warn before attempting to load a model that won't fit in VRAM
    2. Recommend batch size given available VRAM
    3. Log memory usage at key checkpoints during training

Formula for training VRAM:
    model_weights:     params × bytes_per_param
    gradients:         trainable_params × 4  (fp32 gradients)
    optimizer_states:  trainable_params × 8  (AdamW: momentum + RMS in fp32)
    activations:       batch × seq_len × hidden × n_layers × 2 bytes  (rough estimate)
    KV_cache:          0 during training (use_cache=False)

Formula for inference VRAM:
    model_weights:     params × bytes_per_param
    KV_cache:          2 × n_layers × n_heads × head_dim × seq_len × batch × bytes
    activations:       1 layer at a time (not all stored simultaneously)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)

_BYTES_PER_DTYPE: dict[str, float] = {
    "float32":  4.0,
    "bfloat16": 2.0,
    "float16":  2.0,
    "int8":     1.0,
    "nf4":      0.5,   # 4 bits = 0.5 bytes
    "int4":     0.5,
}

# Model size reference table (approximate, varies by exact variant)
_MODEL_SIZE_REFERENCE: dict[str, dict] = {
    "qwen2.5-0.5b":  {"params_B": 0.5,  "layers": 24, "hidden": 1024, "heads": 16},
    "qwen2.5-1.5b":  {"params_B": 1.5,  "layers": 28, "hidden": 1536, "heads": 12},
    "qwen2.5-3b":    {"params_B": 3.0,  "layers": 36, "hidden": 2048, "heads": 16},
    "qwen2.5-7b":    {"params_B": 7.6,  "layers": 32, "hidden": 3584, "heads": 28},
    "mistral-7b":    {"params_B": 7.3,  "layers": 32, "hidden": 4096, "heads": 32},
    "phi-3-mini":    {"params_B": 3.8,  "layers": 32, "hidden": 3072, "heads": 32},
    "llama-3-8b":    {"params_B": 8.0,  "layers": 32, "hidden": 4096, "heads": 32},
}


def estimate_training_vram(
    params_B: float,
    trainable_params_M: float,
    batch_size: int = 4,
    seq_len: int = 2048,
    hidden_size: int = 4096,
    n_layers: int = 32,
    dtype: str = "nf4",
    gradient_checkpointing: bool = True,
) -> dict:
    """
    Estimate total VRAM required for one training step.

    Args:
        params_B:              Total model parameters in billions.
        trainable_params_M:    Trainable parameters (LoRA) in millions.
        batch_size:            Per-device batch size.
        seq_len:               Maximum sequence length.
        hidden_size:           Model hidden dimension.
        n_layers:              Number of transformer layers.
        dtype:                 Model weight dtype ("nf4", "bfloat16", etc.).
        gradient_checkpointing: If True, only ~sqrt(n_layers) activations
                                are stored (reduces activation memory ~5×).

    Returns:
        Dict with per-component VRAM estimates and total in GB.
    """
    bpp = _BYTES_PER_DTYPE.get(dtype, 2.0)

    # Model weights
    model_gb = params_B * 1e9 * bpp / 1e9

    # Gradients (only for trainable params, always fp32)
    grad_gb = trainable_params_M * 1e6 * 4 / 1e9

    # AdamW optimizer states (momentum + RMS, fp32 for trainable params)
    # paged_adamw stores these in CPU RAM, so set to 0 for QLoRA
    optim_gb = trainable_params_M * 1e6 * 8 / 1e9

    # Activations (rough estimate)
    # With gradient checkpointing: only sqrt(n_layers) activations stored
    effective_layers = int(n_layers ** 0.5) if gradient_checkpointing else n_layers
    activ_gb = (
        batch_size * seq_len * hidden_size * effective_layers * 2  # bf16
    ) / 1e9

    total_gb = model_gb + grad_gb + optim_gb + activ_gb
    # Add 10% overhead for PyTorch CUDA memory allocator fragmentation
    total_with_overhead = total_gb * 1.10

    result = {
        "model_weights_gb":         round(model_gb, 2),
        "gradients_gb":             round(grad_gb, 3),
        "optimizer_states_gb":      round(optim_gb, 3),
        "activations_gb":           round(activ_gb, 2),
        "estimated_total_gb":       round(total_gb, 2),
        "with_overhead_gb":         round(total_with_overhead, 2),
        "gradient_checkpointing":   gradient_checkpointing,
        "dtype":                    dtype,
        "batch_size":               batch_size,
        "seq_len":                  seq_len,
    }

    logger.info(
        f"[Memory] Training VRAM estimate ({dtype}): "
        f"weights={model_gb:.1f} + grads={grad_gb:.2f} + "
        f"optim={optim_gb:.2f} + activ={activ_gb:.2f} "
        f"= {total_with_overhead:.1f} GB (with overhead)"
    )
    return result


def estimate_inference_vram(
    params_B: float,
    dtype: str = "bfloat16",
    batch_size: int = 1,
    seq_len: int = 2048,
    n_layers: int = 32,
    n_heads: int = 32,
    head_dim: int = 128,
) -> dict:
    """
    Estimate VRAM required for inference with KV cache.

    The KV cache stores key and value tensors for all tokens in the
    current sequence across all layers and heads. This allows generation
    to avoid recomputing attention for previous tokens.

    KV cache size: 2 (K+V) × layers × heads × head_dim × seq_len × batch × 2 bytes
    """
    bpp = _BYTES_PER_DTYPE.get(dtype, 2.0)
    model_gb = params_B * 1e9 * bpp / 1e9

    # KV cache
    kv_cache_gb = (
        2 * n_layers * n_heads * head_dim * seq_len * batch_size * 2  # bfloat16
    ) / 1e9

    total_gb = model_gb + kv_cache_gb
    result = {
        "model_weights_gb": round(model_gb, 2),
        "kv_cache_gb":      round(kv_cache_gb, 3),
        "total_gb":         round(total_gb, 2),
        "dtype":            dtype,
        "seq_len":          seq_len,
    }

    logger.info(
        f"[Memory] Inference VRAM estimate: "
        f"weights={model_gb:.1f} + kv_cache={kv_cache_gb:.2f} "
        f"= {total_gb:.2f} GB"
    )
    return result


def log_current_vram() -> dict:
    """
    Return current GPU VRAM usage from torch.cuda.

    Returns empty dict on CPU-only machines.
    Call this at key points (after model load, start of training, etc.)
    to track memory growth.
    """
    if not torch.cuda.is_available():
        return {}

    stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        stats[f"gpu_{i}"] = {
            "allocated_gb": round(allocated, 3),
            "reserved_gb":  round(reserved, 3),
            "free_gb":      round(total - reserved, 3),
            "total_gb":     round(total, 1),
            "utilisation_pct": round(100 * reserved / total, 1),
        }

    # Flatten for single-GPU case (most common)
    if len(stats) == 1:
        return stats.get("gpu_0", {})
    return stats


def recommend_batch_size(
    available_vram_gb: float,
    model_vram_gb: float,
    seq_len: int = 2048,
    hidden_size: int = 4096,
) -> int:
    """
    Recommend a per-device batch size given available VRAM.

    Simple heuristic: activation memory scales linearly with batch size.
    We use the remaining VRAM after model loading for activations.

    Args:
        available_vram_gb: Total VRAM on the GPU.
        model_vram_gb:     VRAM consumed by model weights.
        seq_len:           Target sequence length.
        hidden_size:       Model hidden dimension.

    Returns:
        Recommended batch size (minimum 1).
    """
    remaining_gb = max(0, available_vram_gb - model_vram_gb - 1.0)  # 1GB buffer

    # Activation memory per sample per layer (rough): seq * hidden * 2 bytes * ~4 tensors
    bytes_per_sample = seq_len * hidden_size * 2 * 4
    gb_per_sample = bytes_per_sample / 1e9

    if gb_per_sample <= 0:
        return 1

    batch = max(1, int(remaining_gb / gb_per_sample))
    # Round down to power of 2 for hardware efficiency
    power_of_2 = 1
    while power_of_2 * 2 <= batch:
        power_of_2 *= 2

    logger.info(
        f"[Memory] Recommended batch size: {power_of_2} "
        f"(remaining VRAM: {remaining_gb:.1f} GB, "
        f"per-sample: {gb_per_sample:.3f} GB)"
    )
    return power_of_2