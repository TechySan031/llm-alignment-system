"""
Model and tokenizer loading for the LLM Alignment System.

Handles four loading scenarios:
    1. Training (local CPU):   Full precision float32, no quantization.
                               Use a small model like Qwen2.5-0.5B for testing.
    2. Training (cloud GPU):   QLoRA NF4 4-bit quantization via bitsandbytes.
                               Use Qwen2.5-7B-Instruct.
    3. Inference (any device): use_cache=True, padding_side="left".
    4. Adapter loading:        Load base model + saved PEFT adapter.
    5. Merge and save:         Merge LoRA into base for vLLM deployment.

Key implementation notes:

use_cache=False during training:
    The KV cache stores computed key/value attention tensors to avoid
    recomputing them during autoregressive generation. During teacher-forced
    SFT training, we compute the full sequence in one forward pass — we never
    generate token by token. The cache wastes VRAM without any benefit.
    Always re-enable it for inference (use_cache=True).

device_map="auto" with quantization:
    HuggingFace Accelerate maps model layers across available devices.
    With a single GPU, all layers go to cuda:0.
    With multi-GPU, layers are distributed by parameter count.
    With CPU fallback, it uses CPU offloading for layers that don't fit in VRAM.
    On local AMD machine with no CUDA: device_map="cpu" is used instead.

attn_implementation:
    "flash_attention_2" — fastest, requires flash-attn package + Ampere+ GPU
    "sdpa"             — PyTorch scaled_dot_product_attention, good fallback
    "eager"            — standard PyTorch attention, always works including CPU
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
# pyrefly: ignore [missing-import]
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, PreTrainedModel

# pyrefly: ignore [missing-import]
from src.models.quantization import get_bnb_config, get_compute_dtype, is_quantization_available
# pyrefly: ignore [missing-import]
from src.models.tokenizer_loader import load_tokenizer
# pyrefly: ignore [missing-import]
from src.models.parameter_counter import count_parameters
# pyrefly: ignore [missing-import]
from src.models.memory_estimator import log_current_vram
# pyrefly: ignore [missing-import]
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def _get_attn_implementation(cfg: DictConfig) -> str:
    """
    Select attention implementation based on hardware availability.

    Priority:
        1. Config value if explicitly set and hardware supports it
        2. flash_attention_2 if GPU + flash-attn installed
        3. sdpa if GPU present (PyTorch built-in)
        4. eager for CPU (always works)
    """
    config_impl = cfg.model.get("attn_implementation", "auto")

    if config_impl != "auto":
        # User explicitly specified — honour it but warn if unsupported
        if config_impl == "flash_attention_2":
            try:
                # pyrefly: ignore [missing-import]
                import flash_attn  # noqa: F401
                if torch.cuda.is_available():
                    return "flash_attention_2"
                else:
                    logger.warning(
                        "[Loader] flash_attention_2 requested but CUDA unavailable. "
                        "Falling back to eager."
                    )
                    return "eager"
            except ImportError:
                logger.warning(
                    "[Loader] flash_attention_2 requested but flash-attn not installed. "
                    "Install with: pip install flash-attn --no-build-isolation. "
                    "Falling back to sdpa."
                )
                return "sdpa" if torch.cuda.is_available() else "eager"
        return config_impl

    # Auto-select
    if not torch.cuda.is_available():
        return "eager"

    try:
        # pyrefly: ignore [missing-import]
        import flash_attn  # noqa: F401
        major = torch.cuda.get_device_properties(0).major
        if major >= 8:  # Ampere+
            return "flash_attention_2"
    except ImportError:
        pass

    return "sdpa"


def load_base_model(
    cfg: DictConfig,
) -> Tuple[PreTrainedModel, object]:
    """
    Load the base model and tokenizer with hardware-appropriate configuration.

    On local Windows/AMD machine (no CUDA):
        - Loads in float32 on CPU
        - No quantization
        - Uses eager attention
        - Suitable for architecture inspection, unit tests, and small model runs

    On cloud GPU (Colab / RunPod / Kaggle with CUDA):
        - Loads with NF4 4-bit quantization (QLoRA)
        - Uses bfloat16 compute dtype
        - Uses flash_attention_2 if available
        - Suitable for full SFT and DPO training on 7B models

    Args:
        cfg: Hydra DictConfig with model, tokenizer, and quantization sections.

    Returns:
        (model, tokenizer) tuple.
        model is a PreTrainedModel with requires_grad=False on all parameters
        (LoRA injection in lora_config.py adds the trainable adapters).

    Raises:
        OSError: If the model name/path cannot be resolved from HuggingFace Hub
                 or local disk. Check HF_TOKEN in .env for gated models.
    """
    model_name = cfg.model.name
    logger.info(f"[Loader] Loading model: {model_name}")
    logger.info(
        f"[Loader] Hardware: "
        f"{'CUDA GPU' if torch.cuda.is_available() else 'CPU (no GPU)'}"
    )

    # ── Quantization config ───────────────────────────────────────────────────
    bnb_config = None
    if cfg.quantization.get("load_in_4bit", False):
        bnb_config = get_bnb_config(
            load_in_4bit=True,
            compute_dtype=cfg.quantization.get("bnb_4bit_compute_dtype", "bfloat16"),
            quant_type=cfg.quantization.get("bnb_4bit_quant_type", "nf4"),
            double_quant=cfg.quantization.get("bnb_4bit_use_double_quant", True),
        )
        if bnb_config is None:
            logger.info(
                "[Loader] 4-bit quantization requested but CUDA unavailable. "
                "Loading in float32 on CPU for local development."
            )

    # ── dtype and device_map ─────────────────────────────────────────────────
    if bnb_config is not None:
        # Quantized models: dtype is set inside BitsAndBytesConfig
        torch_dtype = None
        device_map = "auto"
    elif torch.cuda.is_available():
        torch_dtype = _DTYPE_MAP.get(cfg.model.get("dtype", "bfloat16"), torch.bfloat16)
        device_map = "auto"
    else:
        # CPU-only local machine
        torch_dtype = torch.float32
        device_map = "cpu"
        logger.info("[Loader] CPU mode: loading in float32")

    # ── Attention implementation ──────────────────────────────────────────────
    attn_impl = _get_attn_implementation(cfg)
    logger.info(f"[Loader] Attention implementation: {attn_impl}")

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=cfg.model.get("trust_remote_code", True),
            use_cache=False,  # Disabled during training — see module docstring
            attn_implementation=attn_impl,
        )
    except Exception as e:
        _handle_load_error(e, model_name)
        raise

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(
        model_name=model_name,
        padding_side=cfg.tokenizer.get("padding_side", "right"),
        max_length=cfg.tokenizer.get("max_length", 2048),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
    )

    # ── Log diagnostics ───────────────────────────────────────────────────────
    stats = count_parameters(model)
    logger.info(
        f"[Loader] Parameters — "
        f"total: {stats['total_B']:.3f}B | "
        f"trainable: {stats['trainable_M']:.2f}M | "
        f"trainable%: {stats['trainable_pct']:.4f}%"
    )

    vram = log_current_vram()
    if vram:
        logger.info(
            f"[Loader] VRAM after load — "
            f"allocated: {vram.get('allocated_gb', 0):.2f} GB | "
            f"reserved: {vram.get('reserved_gb', 0):.2f} GB"
        )

    return model, tokenizer


def load_model_for_inference(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> Tuple[PreTrainedModel, object]:
    """
    Load a model in inference mode.

    Differences from load_base_model:
        - use_cache=True  (KV cache enabled for fast generation)
        - padding_side="left"  (required for batched generation)
        - No quantization config (full precision for inference quality)
        - model.eval() called before returning

    Args:
        model_name_or_path: HuggingFace model name or local path to
                            a merged model directory.
        dtype:              Compute dtype string.
        device_map:         Device placement strategy.

    Returns:
        (model, tokenizer) ready for generation.
    """
    logger.info(f"[Loader] Loading for inference: {model_name_or_path}")

    if not torch.cuda.is_available():
        torch_dtype = torch.float32
        device_map = "cpu"
        logger.info("[Loader] CPU inference mode")
    else:
        torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_cache=True,  # Enable KV cache for inference
        )
        model.eval()
    except Exception as e:
        _handle_load_error(e, model_name_or_path)
        raise

    tokenizer = load_tokenizer(
        model_name=model_name_or_path,
        padding_side="left",  # Left-pad for batched generation
    )

    stats = count_parameters(model)
    logger.info(
        f"[Loader] Inference model loaded — {stats['total_B']:.3f}B params"
    )
    return model, tokenizer


def load_peft_model(
    base_model: PreTrainedModel,
    adapter_path: str,
    is_trainable: bool = False,
) -> object:
    """
    Load a saved LoRA adapter onto an existing base model.

    Args:
        base_model:   A model loaded with load_base_model().
        adapter_path: Path to the saved PEFT adapter directory
                      (contains adapter_config.json and adapter_model.bin).
        is_trainable: True for continued training (DPO after SFT).
                      False for inference evaluation.

    Returns:
        PeftModel wrapping the base model with the loaded adapter.

    Example:
        base_model, tokenizer = load_base_model(cfg)
        model = load_peft_model(base_model, "experiments/sft_runs/run_001/final_adapter")
    """
    # pyrefly: ignore [missing-import]
    from peft import PeftModel

    adapter_path = str(adapter_path)
    if not Path(adapter_path).exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_path}\n"
            "Run scripts/train_sft.py first to generate an adapter."
        )

    logger.info(f"[Loader] Loading PEFT adapter: {adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=is_trainable,
    )

    if not is_trainable:
        model.eval()

    stats = count_parameters(model)
    logger.info(
        f"[Loader] Adapter loaded — "
        f"trainable: {stats['trainable_M']:.2f}M ({stats['trainable_pct']:.4f}%)"
    )
    return model


def merge_and_save_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    dtype: str = "bfloat16",
) -> None:
    """
    Merge LoRA adapter weights into the base model and save the result.

    This produces a standalone model with no PEFT dependency:
        W_merged = W_frozen + (alpha / r) * B @ A

    When to use this:
        - Before deploying with vLLM (vLLM loads merged models)
        - Before uploading to HuggingFace Hub
        - When you want a single model file for distribution

    When NOT to use this:
        - During training (merge in-place prevents further adapter updates)
        - When you need to switch between multiple adapters on one base model

    Memory requirement: fits both base model and adapter in VRAM simultaneously.
    For 7B bfloat16: ~28 GB VRAM. Use a high-memory instance for this step.

    Args:
        base_model_name: Original base model name (must match what was fine-tuned).
        adapter_path:    Path to saved PEFT adapter.
        output_path:     Where to save the merged model.
        dtype:           Precision for the saved merged weights.
    """
    logger.info(
        f"[Loader] Merging adapter: {adapter_path} → {output_path}"
    )

    torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)

    # Load base at full precision for a clean merge
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    # pyrefly: ignore [missing-import]
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_path, safe_serialization=True)

    # Save tokenizer alongside merged model
    tokenizer = load_tokenizer(base_model_name)
    tokenizer.save_pretrained(output_path)

    # pyrefly: ignore [missing-import]
    from src.utils.file_utils import get_dir_size_mb
    size_mb = get_dir_size_mb(output_path)
    logger.info(
        f"[Loader] Merged model saved → {output_path} "
        f"({size_mb / 1000:.1f} GB)"
    )


def _handle_load_error(error: Exception, model_name: str) -> None:
    """Provide actionable error messages for common loading failures."""
    err_str = str(error).lower()

    if "gated" in err_str or "401" in err_str or "unauthorized" in err_str:
        logger.error(
            f"[Loader] Authentication error loading {model_name}.\n"
            "This model requires a HuggingFace token.\n"
            "Solution: Set HF_TOKEN in your .env file and run:\n"
            "  huggingface-cli login"
        )
    elif "out of memory" in err_str or "oom" in err_str:
        logger.error(
            f"[Loader] Out of memory loading {model_name}.\n"
            "Solutions:\n"
            "  1. Enable 4-bit quantization: quantization.load_in_4bit=true\n"
            "  2. Use a smaller model: model.name=Qwen/Qwen2.5-0.5B-Instruct\n"
            "  3. Use gradient_checkpointing=true in training config\n"
            "  4. Reduce per_device_train_batch_size"
        )
    elif "not found" in err_str or "does not exist" in err_str:
        logger.error(
            f"[Loader] Model not found: {model_name}.\n"
            "Check the model name at https://huggingface.co/models\n"
            "Or verify the local path exists."
        )
    elif "no module named 'bitsandbytes'" in err_str:
        logger.error(
            "[Loader] bitsandbytes not installed.\n"
            "Install with: pip install bitsandbytes\n"
            "Note: bitsandbytes requires a CUDA GPU. "
            "On local CPU machines, set quantization.load_in_4bit=false"
        )
    else:
        logger.error(f"[Loader] Model load failed: {error}")