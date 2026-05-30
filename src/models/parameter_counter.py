"""
Trainable parameter counting and LoRA efficiency reporting.

Used after prepare_model_for_training() to verify LoRA injection
produced the expected parameter counts and to generate the
"trainable parameters" line that appears in the final README.

Key metric: trainable_pct
    For QLoRA r=16 on Qwen2.5-7B targeting all projections:
        Total:     ~7.6 billion
        Trainable: ~40 million
        Trainable: 0.53%

    This is the central QLoRA result: <1% of parameters are trained,
    yet the model achieves near full fine-tuning quality.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import torch

# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)


def count_parameters(model) -> dict:
    """
    Count and categorise all model parameters.

    Returns:
        Dict with keys:
            total_B:        Total parameters in billions (float)
            trainable_M:    Trainable parameters in millions (float)
            frozen_B:       Frozen parameters in billions (float)
            trainable_pct:  Percentage of trainable parameters (float)
            by_module:      Dict of {module_name: {total, trainable}} (dict)

    Example:
        stats = count_parameters(model)
        print(f"Trainable: {stats['trainable_M']:.1f}M ({stats['trainable_pct']:.2f}%)")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    by_module: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "trainable": 0})
    for name, param in model.named_parameters():
        # Use top-level module name as the group key
        top_level = name.split(".")[0]
        by_module[top_level]["total"] += param.numel()
        if param.requires_grad:
            by_module[top_level]["trainable"] += param.numel()

    return {
      "total_B": total / 1e9,
    "trainable_M": trainable / 1e6,
    "frozen_B": frozen / 1e9,
    "trainable_pct": 100.0 * trainable / max(total, 1),
    "by_module": dict(by_module),
}

def count_lora_parameters(model) -> dict:
    """
    Count LoRA-specific parameters (A matrices, B matrices separately).

    Useful for verifying:
        - B matrices are zero at the start of training (ΔW = 0)
        - A and B dimensions match the configured rank r

    Returns:
        Dict with lora_A_params, lora_B_params, total_lora_params, n_lora_layers.
    """
    lora_A_params = 0
    lora_B_params = 0
    n_lora_layers = 0

    for name, param in model.named_parameters():
        if "lora_A" in name:
            lora_A_params += param.numel()
            n_lora_layers += 1
        elif "lora_B" in name:
            lora_B_params += param.numel()

    return {
        "lora_A_params_M": round(lora_A_params / 1e6, 3),
        "lora_B_params_M": round(lora_B_params / 1e6, 3),
        "total_lora_M":    round((lora_A_params + lora_B_params) / 1e6, 3),
        "n_lora_layers":   n_lora_layers,
    }


def print_parameter_table(model) -> None:
    """
    Print a formatted table of parameters grouped by top-level module.

    Output example:
        ──────────────────────────────────────────────────────────────
          Module                            Total        Trainable
        ──────────────────────────────────────────────────────────────
          model                         7,563.00M        40.21M
          lm_head                          32.00M         0.00M
        ──────────────────────────────────────────────────────────────
          TOTAL                             7.595B        40.21M
          Trainable:  0.5306%
        ──────────────────────────────────────────────────────────────
    """
    stats = count_parameters(model)
    bar = "─" * 62
    print(f"\n{bar}")
    print(f"  {'Module':<34} {'Total':>12} {'Trainable':>12}")
    print(f"{bar}")
    for mod, s in sorted(stats["by_module"].items()):
        t = s["total"] / 1e6
        r = s["trainable"] / 1e6
        print(f"  {mod:<34} {t:>10.2f}M  {r:>10.2f}M")
    print(f"{bar}")
    print(
        f"  {'TOTAL':<34} {stats['total_B']:>10.3f}B  "
        f"{stats['trainable_M']:>10.3f}M"
    )
    print(f"  Trainable: {stats['trainable_pct']:.5f}%")
    print(f"{bar}\n")


def lora_efficiency_report(model) -> dict:
    """
    Generate a parameter efficiency report comparing LoRA to full fine-tuning.

    Computes:
        efficiency_ratio:  How many times fewer parameters LoRA trains vs full FT
        param_savings_pct: Percentage of parameter updates saved
        grad_memory_saved_gb: Estimated GPU memory saved (fp32 gradients)

    Returns dict suitable for logging to W&B config.
    """
    stats = count_parameters(model)
    total_M = stats["total_B"] * 1000
    trainable_M = stats["trainable_M"]
    efficiency = total_M / max(trainable_M, 0.001)
    savings_pct = 100.0 * (1.0 - trainable_M / max(total_M, 1))

    # fp32 gradient memory: 4 bytes per parameter
    full_ft_grad_gb = total_M * 1e6 * 4 / 1e9
    lora_grad_gb = trainable_M * 1e6 * 4 / 1e9
    saved_gb = full_ft_grad_gb - lora_grad_gb

    report = {
        "total_params_M":          round(total_M, 1),
        "trainable_params_M":      trainable_M,
        "efficiency_ratio":        round(efficiency, 1),
        "param_savings_pct":       round(savings_pct, 2),
        "full_ft_grad_memory_gb":  round(full_ft_grad_gb, 2),
        "lora_grad_memory_gb":     round(lora_grad_gb, 3),
        "grad_memory_saved_gb":    round(saved_gb, 2),
    }

    logger.info(
        f"[LoRA Efficiency] "
        f"{trainable_M:.1f}M / {total_M:.0f}M params trained | "
        f"{efficiency:.0f}× reduction | "
        f"saves {saved_gb:.1f} GB gradient memory"
    )
    return report


def verify_lora_initialization(model) -> bool:
    """
    Verify that all LoRA B matrices are zero at initialization.

    ΔW = (alpha/r) * B @ A = 0 when B = 0.
    This ensures training starts from exact pretrained model behaviour.
    If this check fails, something went wrong with LoRA injection.

    Returns True if all B matrices are zero (correct), False otherwise.
    """
    all_zero = True
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            if not torch.all(param.data == 0):
                logger.warning(
                    f"[LoRA] B matrix not zero at init: {name} "
                    f"(max abs: {param.data.abs().max().item():.6f})"
                )
                all_zero = False

    if all_zero:
        logger.info("[LoRA] B matrix initialization check passed (all zeros)")
    return all_zero