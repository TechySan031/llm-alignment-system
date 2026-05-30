# pyrefly: ignore [missing-import]
"""
Phase 2 setup and hardware diagnostic script.

Run this before any training to verify:
    - Python and package versions
    - GPU availability and VRAM
    - bitsandbytes quantization availability
    - Tokenizer compatibility
    - VRAM requirements for target model

Usage:
    python scripts/setup_environment.py
    python scripts/setup_environment.py --model Qwen/Qwen2.5-0.5B-Instruct

Expected output (local Windows AMD machine):
    System check passed
    GPU: not available (CPU only)
    Quantization: disabled (no CUDA)
    Recommended model for local testing: Qwen/Qwen2.5-0.5B-Instruct
    Estimated local VRAM needed: 0.0 GB (CPU mode)
    Ready for Phase 3

Expected output (Colab A100):
    System check passed
    GPU: NVIDIA A100-SXM4-40GB | 40.0 GB VRAM | CC 8.0
    Quantization: enabled (NF4 4-bit)
    Recommended compute dtype: bfloat16
    7B model NF4 VRAM estimate: 3.8 GB weights + overhead = ~6 GB
    Ready for Phase 5 (SFT training)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils.logging import setup_logging, log_section, log_dict
from src.utils.file_utils import ensure_dir
from src.models.quantization import (
    is_quantization_available,
    recommend_compute_dtype,
    get_bnb_config,
)
from src.models.memory_estimator import (
    estimate_training_vram,
    log_current_vram,
    recommend_batch_size,
)
from src.utils.reproducibility import log_system_info, get_device


def check_package_versions() -> dict[str, str]:
    """Verify all required packages are installed and return versions."""
    packages = {
        "torch":          "torch",
        "transformers":   "transformers",
        "peft":           "peft",
        "trl":            "trl",
        "datasets":       "datasets",
        "accelerate":     "accelerate",
        "bitsandbytes":   "bitsandbytes",
        "wandb":          "wandb",
        "omegaconf":      "omegaconf",
        "pydantic":       "pydantic",
        "faker":          "faker",
        "fastapi":        "fastapi",
    }
    versions: dict[str, str] = {}
    missing: list[str] = []

    for name, import_name in packages.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "installed")
            versions[name] = version
        except ImportError:
            versions[name] = "MISSING"
            missing.append(name)

    return versions, missing


def check_gpu() -> dict:
    """Return GPU diagnostics."""
    if not torch.cuda.is_available():
        return {"available": False, "device": "CPU", "vram_gb": 0}

    results = {"available": True, "gpus": []}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        results["gpus"].append({
            "id":       i,
            "name":     props.name,
            "vram_gb":  round(props.total_memory / 1e9, 1),
            "cc":       f"{props.major}.{props.minor}",
            "sms":      props.multi_processor_count,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Environment diagnostic for LLM Alignment System")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to check VRAM requirements for"
    )
    parser.add_argument(
        "--test-tokenizer", action="store_true",
        help="Download and test the tokenizer (requires internet)"
    )
    args = parser.parse_args()

    logger = setup_logging(level="INFO", run_name="setup_check")
    ensure_dir("outputs/logs/training")

    log_section(logger, "LLM Alignment System — Environment Check")

    # ── Package versions ──────────────────────────────────────────────────────
    log_section(logger, "Package versions")
    versions, missing = check_package_versions()
    log_dict(logger, versions, "Installed packages")

    if missing:
        logger.warning(
            f"Missing packages: {missing}\n"
            "Install with: pip install -r requirements.txt"
        )
    else:
        logger.info("All required packages installed")

    # ── System info ───────────────────────────────────────────────────────────
    log_section(logger, "System information")
    system_info = log_system_info()

    # ── GPU diagnostics ───────────────────────────────────────────────────────
    log_section(logger, "GPU / compute diagnostics")
    gpu_info = check_gpu()

    if gpu_info["available"]:
        for gpu in gpu_info.get("gpus", []):
            logger.info(
                f"GPU {gpu['id']}: {gpu['name']} | "
                f"{gpu['vram_gb']} GB VRAM | "
                f"CC {gpu['cc']} | {gpu['sms']} SMs"
            )
        vram = log_current_vram()
        log_dict(logger, vram, "Current VRAM usage")
    else:
        logger.info(
            "No CUDA GPU detected.\n"
            "This is correct for local Windows/AMD development.\n"
            "Use Qwen/Qwen2.5-0.5B-Instruct for local testing.\n"
            "Use a cloud GPU (Colab/RunPod) for 7B model training."
        )

    # ── Quantization ──────────────────────────────────────────────────────────
    log_section(logger, "Quantization availability")
    quant_available = is_quantization_available()
    logger.info(
        f"4-bit quantization (bitsandbytes): "
        f"{'ENABLED' if quant_available else 'DISABLED (no CUDA)'}"
    )
    if quant_available:
        compute_dtype = recommend_compute_dtype()
        logger.info(f"Recommended compute dtype: {compute_dtype}")

    # ── VRAM requirements ──────────────────────────────────────────────────────
    log_section(logger, f"VRAM requirements for {args.model}")

    is_7b = "7b" in args.model.lower()
    is_small = "0.5b" in args.model.lower() or "3b" in args.model.lower()
    params_B = 7.6 if is_7b else (0.5 if "0.5b" in args.model.lower() else 3.0)

    if gpu_info["available"] and quant_available:
        estimate = estimate_training_vram(
            params_B=params_B, trainable_params_M=40.0 if is_7b else 5.0,
            batch_size=4, seq_len=2048, dtype="nf4",
        )
        log_dict(logger, estimate, "QLoRA training VRAM estimate")

        vram_gb = gpu_info["gpus"][0]["vram_gb"] if gpu_info.get("gpus") else 0
        if estimate["with_overhead_gb"] > vram_gb:
            logger.warning(
                f"Estimated VRAM ({estimate['with_overhead_gb']:.1f} GB) "
                f"exceeds GPU VRAM ({vram_gb} GB).\n"
                "Reduce batch_size or sequence length."
            )
        else:
            batch = recommend_batch_size(vram_gb, estimate["model_weights_gb"])
            logger.info(
                f"Model fits in GPU VRAM. "
                f"Recommended per_device_train_batch_size: {batch}"
            )
    else:
        logger.info(
            "CPU mode — VRAM requirements do not apply.\n"
            f"For local testing use: Qwen/Qwen2.5-0.5B-Instruct\n"
            f"  float32 memory requirement: ~2 GB RAM\n"
            "For cloud training:\n"
            f"  {args.model} with QLoRA NF4: ~6 GB VRAM\n"
            "  Runs on: T4 (16GB), RTX 3090 (24GB), A100 (40/80GB)"
        )

    # ── Tokenizer test ────────────────────────────────────────────────────────
    if args.test_tokenizer:
        log_section(logger, "Tokenizer compatibility test")
        try:
            from src.models.tokenizer_loader import load_tokenizer, verify_tokenizer_compatibility
            tokenizer = load_tokenizer(args.model, max_length=2048)
            checks = verify_tokenizer_compatibility(tokenizer, max_length=2048)
            log_dict(logger, checks, "Tokenizer checks")
        except Exception as e:
            logger.error(f"Tokenizer test failed: {e}")

    # ── Final recommendation ──────────────────────────────────────────────────
    log_section(logger, "Summary and next steps")

    if not gpu_info["available"]:
        logger.info(
            "Local machine (Windows AMD): READY for development\n"
            "\n"
            "What you can do locally:\n"
            "  python scripts/generate_dataset.py    (Phase 1 dataset)\n"
            "  python scripts/run_baseline.py        (Phase 3 baseline evaluation)\n"
            "  pytest tests/ -v                      (all tests)\n"
            "\n"
            "For training on Qwen2.5-7B, use one of:\n"
            "  Google Colab (Free T4 or Pro A100):\n"
            "    !git clone your_repo && pip install -e . && python scripts/train_sft.py\n"
            "\n"
            "  Kaggle (2× T4, free):\n"
            "    Add repo as dataset, run scripts/train_sft.py in a notebook\n"
            "\n"
            "  RunPod (RTX 3090 ~$0.30/hr, A100 ~$1.50/hr):\n"
            "    runpod.io → deploy PyTorch pod → git clone → pip install -e .\n"
            "\n"
            "  Vast.ai (cheapest A100 rates):\n"
            "    vastai.com → search 'A100 80GB' → install → train"
        )
    else:
        logger.info("GPU available — ready for cloud/local GPU training")

    logger.info("Environment check complete")


if __name__ == "__main__":
    main()