"""
Transformer architecture inspection and visualisation.

Extracts structural information from a loaded model's config and
weights without running a forward pass. Works completely on CPU
because it reads config attributes and parameter shapes, not tensors.

Used by:
    notebooks/02_transformer_inspection.ipynb
    scripts/monitor_training.py
    Evaluation reports (architecture section)

Key concepts surfaced:
    - Head dimension = hidden_size / num_attention_heads
      Smaller head dim = more heads but each attends less expressively
    - FFN ratio = intermediate_size / hidden_size
      Qwen2.5-7B: 18944 / 3584 ≈ 5.3× (higher than typical 4× for SwiGLU)
    - GQA (Grouped Query Attention): num_key_value_heads < num_attention_heads
      Reduces KV cache memory. Qwen2.5-7B: 28 query heads, 4 KV heads → 7× KV reduction
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from transformers import PreTrainedModel

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


def inspect_architecture(model: PreTrainedModel) -> dict:
    """
    Extract key architectural dimensions from a loaded model.

    Reads from model.config, not from weight tensors, so it is fast
    and requires no GPU. Works before and after LoRA injection.

    Returns:
        Dict with architecture metrics. All keys are present but may be
        None if the model config does not expose that attribute.
    """
    cfg = model.config

    hidden = getattr(cfg, "hidden_size", None)
    n_heads = getattr(cfg, "num_attention_heads", None)
    kv_heads = getattr(cfg, "num_key_value_heads", None)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    intermediate = getattr(cfg, "intermediate_size", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    max_pos = getattr(cfg, "max_position_embeddings", None)

    head_dim = hidden // n_heads if (hidden and n_heads) else None
    ffn_ratio = intermediate / hidden if (intermediate and hidden) else None
    gqa_ratio = n_heads / kv_heads if (n_heads and kv_heads and kv_heads > 0) else None
    params_B = sum(p.numel() for p in model.parameters()) / 1e9

    return {
        "model_type":           getattr(cfg, "model_type", "unknown"),
        "num_layers":           n_layers,
        "hidden_size":          hidden,
        "num_attention_heads":  n_heads,
        "num_key_value_heads":  kv_heads,
        "head_dim":             head_dim,
        "intermediate_size":    intermediate,
        "ffn_ratio":            round(ffn_ratio, 2) if ffn_ratio else None,
        "gqa_ratio":            round(gqa_ratio, 1) if gqa_ratio else None,
        "vocab_size":           vocab_size,
        "max_position_embeddings": max_pos,
        "total_params_B":       round(params_B, 3),
        "rope_theta":           getattr(cfg, "rope_theta", None),
        "attention_bias":       getattr(cfg, "attention_bias", None),
        "tie_word_embeddings":  getattr(cfg, "tie_word_embeddings", None),
        "activation_function":  (
            getattr(cfg, "hidden_act", None)
            or getattr(cfg, "activation_function", None)
        ),
    }


def print_architecture_summary(model: PreTrainedModel) -> None:
    """
    Print a human-readable architecture summary to stdout.

    Example output for Qwen2.5-7B-Instruct:
        ════════════════════════════════════════════════════════════
        Architecture: Qwen2.5-7B-Instruct (qwen2)
        ════════════════════════════════════════════════════════════
          Layers:               32
          Hidden size:          3,584
          Attention heads:      28  (KV heads: 4, GQA ratio: 7.0×)
          Head dimension:       128
          FFN size:             18,944  (ratio: 5.3×, activation: silu)
          Vocabulary:           152,064 tokens
          Max sequence length:  32,768
          RoPE theta:           1,000,000.0
          Total parameters:     7.615B
        ════════════════════════════════════════════════════════════
    """
    info = inspect_architecture(model)
    model_name = getattr(model.config, "_name_or_path", "unknown")
    bar = "═" * 60

    print(f"\n{bar}")
    print(f"  Architecture: {model_name} ({info['model_type']})")
    print(f"{bar}")

    print(f"  {'Layers:':<30} {info['num_layers']}")

    hidden_size = (
        f"{info['hidden_size']:,}"
        if info["hidden_size"] is not None
        else "Unknown"
    )
    print(f"  {'Hidden size:':<30} {hidden_size}")

    if info["gqa_ratio"] and info["gqa_ratio"] > 1:
        print(
            f"  {'Attention heads:':<30} "
            f"{info['num_attention_heads']}  "
            f"(KV heads: {info['num_key_value_heads']}, "
            f"GQA ratio: {info['gqa_ratio']:.1f}×)"
        )
    else:
        print(
            f"  {'Attention heads:':<30} "
            f"{info['num_attention_heads']}"
        )

    print(f"  {'Head dimension:':<30} {info['head_dim']}")

    # SAFE FFN HANDLING
    if info["intermediate_size"] is not None:

        ratio_str = (
            f"{info['ffn_ratio']:.1f}×"
            if info["ffn_ratio"] is not None
            else "unknown"
        )

        print(
            f"  {'FFN size:':<30} "
            f"{info['intermediate_size']:,}  "
            f"(ratio: {ratio_str}, "
            f"activation: {info['activation_function']})"
        )

    else:

        print(
            f"  {'FFN size:':<30} "
            f"Unknown "
            f"(activation: {info['activation_function']})"
        )

    vocab_size = (
        f"{info['vocab_size']:,}"
        if info["vocab_size"] is not None
        else "Unknown"
    )
    print(f"  {'Vocabulary:':<30} {vocab_size} tokens")

    max_seq = (
        f"{info['max_position_embeddings']:,}"
        if info["max_position_embeddings"] is not None
        else "Unknown"
    )
    print(f"  {'Max sequence length:':<30} {max_seq}")

    if info["rope_theta"] is not None:
        print(f"  {'RoPE theta:':<30} {info['rope_theta']:,}")

    print(
        f"  {'Total parameters:':<30} "
        f"{info['total_params_B']:.3f}B"
    )

    print(f"{bar}\n")  
        

def get_layer_shapes(model: PreTrainedModel) -> list[dict]:
    """
    Return the shape of every weight tensor in the model.

    Useful for:
        - Verifying LoRA was injected into the right layers
        - Understanding memory distribution across layers
        - Debugging unexpected parameter shapes

    Returns:
        List of dicts with: name, shape, numel, dtype, trainable, is_lora
    """
    shapes = []
    for name, param in model.named_parameters():
        shapes.append({
            "name":      name,
            "shape":     list(param.shape),
            "numel_M":   round(param.numel() / 1e6, 4),
            "dtype":     str(param.dtype).replace("torch.", ""),
            "trainable": param.requires_grad,
            "is_lora":   ("lora_A" in name or "lora_B" in name),
            "device":    str(param.device),
        })
    return shapes


def get_attention_layer_names(model: PreTrainedModel) -> list[str]:
    """
    Return module names that contain attention projections.

    Used by the research module for targeted gradient analysis.
    """
    attn_keywords = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}
    return [
        name for name, module in model.named_modules()
        if any(kw in name for kw in attn_keywords)
        and hasattr(module, "weight")
    ]


def get_ffn_layer_names(model: PreTrainedModel) -> list[str]:
    """Return module names that contain feed-forward network weights."""
    ffn_keywords = {"gate_proj", "up_proj", "down_proj", "gate_up_proj", "c_fc", "c_proj"}
    return [
        name for name, module in model.named_modules()
        if any(kw in name for kw in ffn_keywords)
        and hasattr(module, "weight")
    ]


def compute_parameter_density(model: PreTrainedModel) -> dict[str, float]:
    """
    Compute what fraction of total parameters are in each component type.

    Breaks down: embedding / attention / ffn / layer_norm / output_head.
    Useful for understanding where memory is spent and where LoRA adds the most value.

    Returns:
        Dict of {component: fraction_of_total_params}
    """
    total = sum(p.numel() for p in model.parameters())
    buckets: dict[str, int] = {
        "embedding":    0,
        "attention":    0,
        "ffn":          0,
        "layer_norm":   0,
        "output_head":  0,
        "other":        0,
    }

    for name, param in model.named_parameters():
        n = param.numel()
        name_lower = name.lower()

        if "embed" in name_lower:
            buckets["embedding"] += n
        elif any(k in name_lower for k in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv"]):
            buckets["attention"] += n
        elif any(k in name_lower for k in ["gate_proj", "up_proj", "down_proj", "ffn", "mlp"]):
            buckets["ffn"] += n
        elif any(k in name_lower for k in ["norm", "ln_"]):
            buckets["layer_norm"] += n
        elif "lm_head" in name_lower or "cls" in name_lower:
            buckets["output_head"] += n
        else:
            buckets["other"] += n

    density = {
        component: round(100.0 * count / max(total, 1), 2)
        for component, count in buckets.items()
    }
    return density