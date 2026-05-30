"""
Dynamic padding data collator for SFT.

Dynamic padding: pad each batch only to the length of its longest sequence,
rather than a fixed global max_length. On variable-length data this cuts
wasted compute by 20-40%.

Alignment: pad to multiples of 8 for Tensor Core efficiency on bf16/fp16.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer


@dataclass
class SFTDataCollator:
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        # Round up to nearest multiple of pad_to_multiple_of
        remainder = max_len % self.pad_to_multiple_of
        if remainder:
            max_len += self.pad_to_multiple_of - remainder

        input_ids, attention_masks, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_masks.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }