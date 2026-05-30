"""
Quantization configuration factory.

Handles the detection and configuration of bitsandbytes NF4 quantization
used in QLoRA training. Gracefully degrades to no quantization when
CUDA is not available (local Windows/AMD development machine).

QLoRA quantization stack:
    BitsAndBytesConfig   — tells the HF loader which quantization to apply
    NF4 (NormalFloat4)   — 4-bit quantization using normal distribution bins
    double_quant         — quantizes the quantization constants (saves ~0.4 bits/param)
    compute_dtype=bf16   — dequantizes to bfloat16 for the forward pass

Memory reduction for Qwen2.5-7B:
    float32  (no quant):  ~28 GB
    bfloat16 (no quant):  ~14 GB
    int8     (LLM.int8):  ~7 GB
    nf4      (QLoRA):     ~4 GB
    nf4+dq   (QLoRA):     ~3.5 GB   ← what we use

NF4 vs INT4:
    INT4 divides the [-1, 1] range into 16 equally spaced bins.
    NF4 places bins according to the quantile function of a standard Normal
    distribution. Since LLM weight matrices follow approximately Normal
    distributions, NF4 retains significantly more information per bit.
    The QLoRA paper showed NF4 matches bfloat16 fine-tuning quality on
    most tasks while INT4 shows measurable degradation.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def is_quantization_available() -> bool:
    """
    Return True if bitsandbytes quantization can be used.

    Requirements:
        - CUDA-capable GPU detected by PyTorch
        - bitsandbytes installed
        - GPU compute capability >= 7.0 (Volta+)

    On local Windows/AMD: always returns False.
    On Colab/RunPod with A100/T4/V100: returns True.
    """
    if not torch.cuda.is_available():
        logger.debug("[Quantization] CUDA not available — quantization disabled")
        return False

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        logger.warning(
            "[Quantization] bitsandbytes not installed. "
            "Install with: pip install bitsandbytes"
        )
        return False

    # Check compute capability (Volta = 7.0 minimum)
    major, minor = torch.cuda.get_device_capability(0)
    cc = major + minor / 10
    if cc < 7.0:
        logger.warning(
            f"[Quantization] GPU compute capability {cc:.1f} < 7.0 "
            "— 4-bit quantization not supported on this GPU"
        )
        return False

    return True


def get_compute_dtype(dtype_str: str = "bfloat16") -> torch.dtype:
    """
    Return the torch.dtype for a string identifier.

    For QLoRA, this is the dtype used during the dequantized forward pass.
    Use bfloat16 on Ampere+ (A100, RTX 30/40xx).
    Use float16 on Volta/Turing (V100, RTX 20xx, T4).
    """
    dtype = _DTYPE_MAP.get(dtype_str.lower())
    if dtype is None:
        logger.warning(
            f"[Quantization] Unknown dtype '{dtype_str}'. "
            f"Valid options: {list(_DTYPE_MAP.keys())}. "
            "Defaulting to bfloat16."
        )
        return torch.bfloat16
    return dtype


def recommend_compute_dtype() -> str:
    """
    Recommend bfloat16 or float16 based on the detected GPU.

    Returns a string matching the YAML config format ("bfloat16" / "float16").
    Ampere+ (compute capability >= 8.0) supports bfloat16 natively.
    Older GPUs should use float16 with loss scaling.
    """
    if not torch.cuda.is_available():
        return "float32"
    major = torch.cuda.get_device_properties(0).major
    return "bfloat16" if major >= 8 else "float16"


def get_bnb_config(
    load_in_4bit: bool = True,
    compute_dtype: str = "bfloat16",
    quant_type: str = "nf4",
    double_quant: bool = True,
) -> Optional[object]:
    """
    Build and return a BitsAndBytesConfig for QLoRA training.

    Returns None if quantization is not available so the caller
    can fall back to full-precision loading without code changes.

    Args:
        load_in_4bit:    Use 4-bit NF4 quantization. Set False for int8.
        compute_dtype:   Dtype for dequantized computation ("bfloat16" / "float16").
        quant_type:      Quantization type: "nf4" (recommended) or "fp4".
        double_quant:    Quantize the quantization constants for extra savings.
                         Saves ~0.4 bits per parameter with negligible quality loss.

    Returns:
        BitsAndBytesConfig instance, or None if CUDA / bitsandbytes unavailable.

    Example:
        bnb_cfg = get_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            quantization_config=bnb_cfg,  # None on local machine → full precision
        )
    """
    if not is_quantization_available():
        logger.info(
            "[Quantization] Returning None — model will load in full precision. "
            "This is correct for local CPU development."
        )
        return None

    from transformers import BitsAndBytesConfig

    dtype = get_compute_dtype(compute_dtype)
    config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
    )

    logger.info(
        f"[Quantization] BitsAndBytesConfig: "
        f"4-bit={load_in_4bit} | "
        f"type={quant_type} | "
        f"compute={compute_dtype} | "
        f"double_quant={double_quant}"
    )
    return config


def get_8bit_config() -> Optional[object]:
    """
    Build an INT8 config using LLM.int8() — less aggressive than NF4.

    INT8 uses mixed-precision decomposition: large outlier features
    are kept in float16, remaining activations quantized to int8.
    Approximately 2× memory reduction vs float16. Use when NF4 is
    too aggressive for your task.
    """
    if not is_quantization_available():
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,  # Outlier threshold from the LLM.int8() paper
    )