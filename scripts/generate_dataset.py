"""
Generate the complete SFT + DPO dataset.

This script runs entirely on the local Windows/AMD machine.
No GPU required. Estimated runtime:
    Synthetic generation only: ~30 seconds
    With public datasets:      ~3-5 minutes (network dependent)

Usage:
    # Full hybrid dataset (synthetic + public)
    python scripts/generate_dataset.py

    # Synthetic only (faster, works offline)
    python scripts/generate_dataset.py --no-public

    # Small dataset for testing
    python scripts/generate_dataset.py --no-public --synthetic-scale 0.1

    # Custom seed
    python scripts/generate_dataset.py --seed 99

Expected outputs:
    data/processed/train.jsonl          SFT training examples (raw SFTExample dicts)
    data/processed/validation.jsonl     SFT validation examples
    data/processed/test.jsonl           Test examples (raw SFTExample dicts)
    data/processed/dpo_train.jsonl      DPO training pairs (prompt/chosen/rejected)
    data/processed/dpo_validation.jsonl DPO validation pairs
    data/processed/metadata.json        DatasetMetadata record
    outputs/reports/training_reports/dataset_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from src.data.generator import DatasetOrchestrator
# pyrefly: ignore [missing-import]
from src.data.schemas import DatasetMetadata, DatasetSplit, SFTExample
# pyrefly: ignore [missing-import]
from src.utils.file_utils import ensure_dir, write_json, write_jsonl
# pyrefly: ignore [missing-import]
from src.utils.logging import setup_logging


def compute_split_stats(examples: list[SFTExample], path: str) -> DatasetSplit:
    split_name = Path(path).stem
    valid_names = {"train", "validation", "test"}
    if split_name not in valid_names:
        split_name = "train"
    return DatasetSplit(
        split_name=split_name,  # type: ignore[arg-type]
        n_examples=len(examples),
        by_task={k: v for k, v in Counter(
            ex.task_type.value for ex in examples
        ).items()},
        by_difficulty={k: v for k, v in Counter(
            ex.difficulty.value for ex in examples
        ).items()},
        by_quality={k: v for k, v in Counter(
            ex.quality.value for ex in examples
        ).items()},
        validity_pct=100.0,
        path=path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT + DPO dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-public", action="store_true",
        help="Skip public dataset loading (offline / faster mode)"
    )
    parser.add_argument(
        "--synthetic-scale", type=float, default=1.0,
        help="Scale synthetic counts by this factor (0.1 for quick tests)"
    )
    parser.add_argument(
        "--ultrachat-n", type=int, default=2000,
        help="Number of UltraChat examples to load"
    )
    parser.add_argument(
        "--openhermes-n", type=int, default=1500,
        help="Number of OpenHermes examples to load"
    )
    args = parser.parse_args()

    logger = setup_logging(
        level="INFO",
        log_dir="outputs/logs/training",
        run_name="generate_dataset",
    )

    ensure_dir("data/processed")
    ensure_dir("outputs/reports/training_reports")

    # ── Step 1: Generate SFT examples ────────────────────────────────────────
    logger.info(
        f"Starting dataset generation — seed={args.seed}, "
        f"public={'disabled' if args.no_public else 'enabled'}"
    )

    orchestrator = DatasetOrchestrator(
        seed=args.seed,
        load_public=not args.no_public,
        ultrachat_n=args.ultrachat_n,
        openhermes_n=args.openhermes_n,
    )

    sft_examples = orchestrator.generate_sft_examples()
    logger.info(f"Total SFT examples generated: {len(sft_examples):,}")

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    import random
    random.seed(args.seed)
    random.shuffle(sft_examples)

    n = len(sft_examples)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = sft_examples[:train_end]
    val = sft_examples[train_end:val_end]
    test = sft_examples[val_end:]

    logger.info(
        f"Split — train: {len(train):,}, val: {len(val):,}, test: {len(test):,}"
    )

    # ── Step 3: Save SFT splits as JSONL ─────────────────────────────────────
    write_jsonl([ex.to_dict() for ex in train], "data/processed/train.jsonl")
    write_jsonl([ex.to_dict() for ex in val], "data/processed/validation.jsonl")
    write_jsonl([ex.to_dict() for ex in test], "data/processed/test.jsonl")
    logger.info("SFT splits saved to data/processed/")

    # ── Step 4: Build DPO pairs ───────────────────────────────────────────────
    # We need a tokenizer to format the prompt with apply_chat_template.
    # For the generation script, we use a lightweight approach:
    # format the prompt manually using the ChatML template string directly.
    # This avoids loading a large model just for dataset generation.
    logger.info("Building DPO pairs (prompt formatting without model load)...")

    dpo_train_pairs: list[dict] = []
    dpo_val_pairs: list[dict] = []

    # pyrefly: ignore [missing-import]
    from src.data.preprocessor import RejectionEngine
    # pyrefly: ignore [missing-import]
    from src.data.schemas import DPOExample, RejectionStrategy

    engine = RejectionEngine()

    def chatML_prompt(ex: SFTExample) -> str:
        """Format prompt using ChatML without loading tokenizer."""
        parts = []
        for turn in ex.conversation.prompt_only:
            parts.append(f"<|im_start|>{turn.role.value}\n{turn.content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    for ex in train:
        reference = ex.conversation.final_response
        if not reference:
            continue
        try:
            rejected, strategy = engine.generate(reference, ex.task_type)
            if rejected.strip() == reference.strip():
                continue
            pair = DPOExample(
                sft_example_id=ex.example_id,
                prompt=chatML_prompt(ex),
                chosen=reference,
                rejected=rejected,
                preference_strength=1.0,
                rejection_strategy=strategy,
                task_type=ex.task_type,
                difficulty=ex.difficulty,
            )
            dpo_train_pairs.append(pair.to_trl_dict())
        except Exception as e:
            logger.debug(f"DPO pair failed: {e}")

    # 85/15 split on DPO pairs
    import random as _r
    _r.shuffle(dpo_train_pairs)
    dpo_split = int(len(dpo_train_pairs) * 0.85)
    dpo_val_pairs = dpo_train_pairs[dpo_split:]
    dpo_train_pairs = dpo_train_pairs[:dpo_split]

    write_jsonl(dpo_train_pairs, "data/processed/dpo_train.jsonl")
    write_jsonl(dpo_val_pairs, "data/processed/dpo_validation.jsonl")
    logger.info(
        f"DPO pairs — train: {len(dpo_train_pairs):,}, "
        f"val: {len(dpo_val_pairs):,}"
    )

    # ── Step 5: Save metadata ─────────────────────────────────────────────────
    metadata = DatasetMetadata(
        total_examples=len(sft_examples),
        splits={
            "train": compute_split_stats(train, "data/processed/train.jsonl"),
            "validation": compute_split_stats(val, "data/processed/validation.jsonl"),
            "test": compute_split_stats(test, "data/processed/test.jsonl"),
        },
        dpo_pairs=len(dpo_train_pairs) + len(dpo_val_pairs),
        seed=args.seed,
        generation_config={
            "load_public": not args.no_public,
            "ultrachat_n": args.ultrachat_n,
            "openhermes_n": args.openhermes_n,
        },
    )
    write_json(metadata.to_dict(), "data/processed/metadata.json")
    write_json(metadata.to_dict(), "outputs/reports/training_reports/dataset_report.json")

    logger.info("Dataset generation complete.")
    logger.info(
        f"\n  Total SFT:  {len(sft_examples):,}\n"
        f"  DPO train:  {len(dpo_train_pairs):,}\n"
        f"  DPO val:    {len(dpo_val_pairs):,}\n"
        f"  Metadata:   data/processed/metadata.json"
    )


if __name__ == "__main__":
    main()