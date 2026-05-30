"""
LoRA and QLoRA adapter configuration and injection.

LoRA mathematics:
    Standard fine-tuning: W_new = W + ΔW
    ΔW has the same shape as W — for a 4096×4096 projection, that is
    16.7M parameters per layer × 32 layers = 536M extra parameters.

    LoRA decomposes ΔW as a product of two low-rank matrices:
        ΔW ≈ B @ A    where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d, k)

    Full forward pass during training:
        h = W_frozen @ x + (α/r) * B @ (A @ x)

    Initialisation:
        A ~ Kaiming uniform (ensures reasonable gradient magnitude at step 0)
        B = 0             (ensures ΔW = 0 at step 0 → training starts from
                           exact pretrained behaviour)

    Effective weight update scale:
        α/r controls how strongly the adapter output is weighted.
        α = 32, r = 16 → scale = 2.0 (adapter contributes 2× per unit LR)
        α = r          → scale = 1.0 (no scaling; effective LR = optimizer LR)

Memory savings for Qwen2.5-7B (r=16, all projections):
    Full fine-tuning:  7.6B parameters × 4 bytes (fp32 grads) = 30.4 GB
    LoRA adapters:     ~40M parameters × 4 bytes (fp32 grads) = 0.16 GB
    Reduction:         ~190× fewer bytes for gradients

Target modules:
    We target all linear projections in both attention and FFN.
    Original LoRA paper targeted only Q, V — sufficient for many NLU tasks
    but suboptimal for complex instruction following and structured output,
    where the FFN layers encode output format patterns.

prepare_model_for_kbit_training:
    Required before LoRA injection when the model is quantized.
    Two effects:
    1. Casts LayerNorm weights to float32. LayerNorm with int4/int8 inputs
       can produce NaN gradients because the variance computation squares
       small values. float32 prevents underflow.
    2. Enables gradient checkpointing. Recomputes activations during the
       backward pass instead of storing them. Saves ~60% activation VRAM
       at the cost of ~30% more compute.
"""
from __future__ import annotations

import logging
from typing import Optional

from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Architecture-specific target module names.
# These are the weight matrices inside each transformer block that LoRA adapts.
# The names come from the actual parameter names in the HuggingFace model.
# Use model.named_modules() to verify for a new architecture.
_TARGET_MODULES: dict[str, list[str]] = {
    "qwen2": [
        "q_proj", "k_proj", "v_proj", "o_proj",    # Attention projections
        "gate_proj", "up_proj", "down_proj",          # FFN projections (SwiGLU)
    ],
    "mistral": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "llama": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "phi3": [
        "qkv_proj", "o_proj",                        # Phi-3 fuses QKV
        "gate_up_proj", "down_proj",                  # Phi-3 fuses gate+up
    ],
    "gemma2": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "gpt2": [
        "c_attn", "c_proj",                          # GPT-2 style (for testing)
        "c_fc",
    ],
    "default": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


def detect_model_family(model_name: str) -> str:
    """
    Detect the model architecture family from its name string.

    Used to select the correct target modules automatically.
    Add new model families here as needed.

    Args:
        model_name: HuggingFace model name or local path.

    Returns:
        Family key matching _TARGET_MODULES dict. "default" if unknown.
    """
    name_lower = model_name.lower()

    if "qwen" in name_lower:
        return "qwen2"
    if "mistral" in name_lower or "mixtral" in name_lower:
        return "mistral"
    if "llama" in name_lower or "vicuna" in name_lower or "alpaca" in name_lower:
        return "llama"
    if "phi-3" in name_lower or "phi3" in name_lower:
        return "phi3"
    if "gemma" in name_lower:
        return "gemma2"
    if "gpt2" in name_lower or "gpt-2" in name_lower:
        return "gpt2"

    logger.warning(
        f"[LoRA] Unknown model family for '{model_name}'. "
        "Using default target modules (LLaMA-style projections). "
        "If training results are poor, manually set lora.target_modules in config."
    )
    return "default"


def build_lora_config(
    cfg: DictConfig,
    model_name: str = "",
) -> LoraConfig:
    """
    Build a LoraConfig from the training YAML config.

    Args:
        cfg:        Hydra DictConfig with a 'lora' section.
        model_name: Used to auto-detect target modules if not set in config.

    Returns:
        LoraConfig ready for get_peft_model().

    Config section example (configs/training/sft_lora.yaml):
        lora:
          r: 16
          lora_alpha: 32
          lora_dropout: 0.05
          bias: "none"
          target_modules:          # Optional: auto-detected if not set
            - "q_proj"
            - "k_proj"
    """
    lora_cfg = cfg.lora

    # Resolve target modules
    if lora_cfg.get("target_modules"):
        target_modules = list(lora_cfg.target_modules)
        logger.info(f"[LoRA] Using config target modules: {target_modules}")
    else:
        family = detect_model_family(model_name)
        target_modules = _TARGET_MODULES.get(family, _TARGET_MODULES["default"])
        logger.info(
            f"[LoRA] Auto-detected family: '{family}' → "
            f"target modules: {target_modules}"
        )

    r = lora_cfg.r
    alpha = lora_cfg.lora_alpha
    scale = alpha / r

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    logger.info(
        f"[LoRA] Config — "
        f"r={r} | alpha={alpha} | scale={scale:.2f}x | "
        f"dropout={lora_cfg.lora_dropout} | bias={lora_cfg.bias} | "
        f"n_target_modules={len(target_modules)}"
    )

    _log_parameter_estimate(r, target_modules, model_name)
    return config


def prepare_model_for_training(
    model,
    cfg: DictConfig,
    is_quantized: bool = True,
) -> object:
    """
    Inject LoRA adapters into the model and prepare for training.

    Two-step process:
        Step 1 (quantized only): prepare_model_for_kbit_training()
            - Casts LayerNorm to float32 (prevents NaN gradients)
            - Enables gradient checkpointing (saves 60% activation VRAM)
            - Only needed when the model was loaded with BitsAndBytesConfig

        Step 2: get_peft_model()
            - Freezes all base model weights (requires_grad=False)
            - Injects trainable LoRA A, B matrices into target layers
            - Wraps in PeftModel that correctly routes gradients

    After this call:
        model.parameters() includes both frozen base params and trainable adapters.
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        → only counts LoRA A and B matrices (~40M for r=16 on 7B model)

    Args:
        model:        Base model loaded with load_base_model().
        cfg:          Hydra DictConfig with 'lora' section.
        is_quantized: True if model was loaded with BitsAndBytesConfig.
                      False for full-precision local CPU training/testing.

    Returns:
        PeftModel with LoRA adapters injected and base weights frozen.
    """
    import torch
    model_name = getattr(model.config, "_name_or_path", "")
    lora_config = build_lora_config(cfg, model_name)

    # Step 1: kbit training prep (only for quantized models on GPU)
    if is_quantized and torch.cuda.is_available():
        from peft import prepare_model_for_kbit_training
        logger.info("[LoRA] Preparing quantized model for kbit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # use_reentrant=False: more memory-efficient and avoids subtle
            # autograd bugs when combined with PEFT
        )
    elif not torch.cuda.is_available():
        logger.info(
            "[LoRA] CPU mode: skipping prepare_model_for_kbit_training "
            "(not needed without quantization)"
        )
        # Still enable gradient checkpointing on CPU for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    # Step 2: inject LoRA adapters
    logger.info("[LoRA] Injecting LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # Verify injection succeeded
    lora_modules = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    if not lora_modules:
        raise RuntimeError(
            "[LoRA] Injection produced 0 adapter modules.\n"
            "This means none of the target_modules matched any layer names.\n"
            f"Configured target_modules: {list(lora_config.target_modules)}\n"
            "Debug: print layer names with:\n"
            "  for name, _ in model.named_modules(): print(name)"
        )

    logger.info(
        f"[LoRA] Injection complete — "
        f"adapters in {len(lora_modules)} modules"
    )

    # Print trainable parameter summary
    model.print_trainable_parameters()
    return model


def get_lora_modules(model) -> list[str]:
    """
    Return the names of all modules that have LoRA adapters injected.

    Useful for verifying that injection hit the expected layers.
    """
    return [name for name, module in model.named_modules() if hasattr(module, "lora_A")]


def compute_lora_delta_weight(model, layer_name: str):
    """
    Compute the effective weight update ΔW = (alpha/r) * B @ A for one layer.

    Used in architecture_viz and research modules to analyse what the
    LoRA adapter actually learned.

    Args:
        model:      PeftModel with trained adapters.
        layer_name: Dot-separated module name (e.g. "model.layers.0.self_attn.q_proj").

    Returns:
        torch.Tensor of shape (out_features, in_features) representing ΔW.
        None if the layer has no LoRA adapter.
    """
    import torch
    module = dict(model.named_modules()).get(layer_name)
    if module is None or not hasattr(module, "lora_A"):
        return None

    try:
        lora_A = module.lora_A["default"].weight.float()  # (r, in_features)
        lora_B = module.lora_B["default"].weight.float()  # (out_features, r)
        scale = module.scaling.get("default", 1.0)
        with torch.no_grad():
            delta_W = scale * (lora_B @ lora_A)
        return delta_W
    except Exception as e:
        logger.debug(f"[LoRA] Could not compute delta weight for {layer_name}: {e}")
        return None


def _log_parameter_estimate(
    r: int,
    target_modules: list[str],
    model_name: str,
) -> None:
    """
    Log a rough estimate of how many parameters the LoRA config will add.

    Uses typical projection dimensions for common model families.
    Actual numbers come from parameter_counter.py after model loading.
    """
    # Rough dimension estimates by model family
    dim_estimates = {
        "7b":   {"d": 4096, "intermediate": 11008},
        "0.5b": {"d": 1024, "intermediate": 2816},
        "3b":   {"d": 2048, "intermediate": 5504},
    }

    size_key = "7b"
    name_lower = model_name.lower()
    if "0.5b" in name_lower or "small" in name_lower:
        size_key = "0.5b"
    elif "3b" in name_lower:
        size_key = "3b"

    dims = dim_estimates.get(size_key, dim_estimates["7b"])
    d = dims["d"]

    # Each attention projection: d×d. Each FFN projection: d×intermediate or vice versa
    attn_modules = [m for m in target_modules if "_proj" in m and m not in
                    ("gate_proj", "up_proj", "down_proj")]
    ffn_modules = [m for m in target_modules if m in ("gate_proj", "up_proj", "down_proj")]

    # LoRA params per module: r*k + d*r = r*(k+d) ≈ 2*r*d (for square matrices)
    params_per_attn_module = 2 * r * d
    params_per_ffn_module = r * (d + dims["intermediate"])

    # Typical 7B has 32 layers
    n_layers = 32
    total_estimate = (
        len(attn_modules) * params_per_attn_module * n_layers
        + len(ffn_modules) * params_per_ffn_module * n_layers
    )

    logger.info(
        f"[LoRA] Estimated trainable params: "
        f"~{total_estimate / 1e6:.1f}M "
        f"({len(target_modules)} modules × {n_layers} layers × 2r×d)"
    )