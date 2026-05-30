"""
Dataset quality validators.

Runs after dataset generation and before preprocessing/training.

Purpose:
- Catch malformed SFT examples
- Catch malformed DPO pairs
- Verify conversations follow schema rules
- Verify JSON structured-output tasks
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from src.data.schemas import (
    Conversation,
    DPOExample,
    Role,
    SFTExample,
    TaskType,
)

logger = logging.getLogger(__name__)


class DatasetValidator:
    def __init__(self):
        self.errors: list[dict] = []

    def validate_sft_example(self, example: SFTExample) -> bool:
        """
        Validate one SFT example.
        """

        try:
            # Conversation exists
            if not isinstance(example.conversation, Conversation):
                raise ValueError("conversation must be Conversation object")

            # Must start with system
            if example.conversation.turns[0].role != Role.SYSTEM:
                raise ValueError("first turn must be system")

            # Must end with assistant
            if example.conversation.turns[-1].role != Role.ASSISTANT:
                raise ValueError("last turn must be assistant")

            # Structured extraction tasks must contain valid JSON
            if example.task_type == TaskType.STRUCTURED_EXTRACTION:
                json.loads(example.target_text)

            return True

        except Exception as e:
            self.errors.append(
                {
                    "example_id": example.example_id,
                    "task_type": example.task_type.value,
                    "error": str(e),
                }
            )
            return False

    def validate_dpo_example(self, example: DPOExample) -> bool:
        """
        Validate one DPO pair.
        """

        try:
            if not example.prompt.strip():
                raise ValueError("empty prompt")

            if not example.chosen.strip():
                raise ValueError("empty chosen response")

            if not example.rejected.strip():
                raise ValueError("empty rejected response")

            if example.chosen.strip() == example.rejected.strip():
                raise ValueError(
                    "chosen and rejected responses are identical"
                )

            return True

        except Exception as e:
            self.errors.append(
                {
                    "pair_id": example.pair_id,
                    "task_type": example.task_type.value,
                    "error": str(e),
                }
            )
            return False

    def validate_sft_dataset(
        self,
        examples: list[SFTExample],
    ) -> dict:
        valid = 0
        invalid = 0

        for ex in examples:
            if self.validate_sft_example(ex):
                valid += 1
            else:
                invalid += 1

        task_counts = Counter(
            ex.task_type.value for ex in examples
        )

        difficulty_counts = Counter(
            ex.difficulty.value for ex in examples
        )

        report = {
            "total": len(examples),
            "valid": valid,
            "invalid": invalid,
            "validity_pct": round(
                100 * valid / max(len(examples), 1),
                2,
            ),
            "task_distribution": dict(task_counts),
            "difficulty_distribution": dict(
                difficulty_counts
            ),
            "first_errors": self.errors[:5],
        }

        if invalid:
            logger.error(
                f"Dataset validation failed "
                f"({invalid} invalid examples)"
            )
        else:
            logger.info(
                f"Dataset validation passed "
                f"({valid} examples)"
            )

        return report

    def validate_dpo_dataset(
        self,
        examples: list[DPOExample],
    ) -> dict:
        valid = 0
        invalid = 0

        for ex in examples:
            if self.validate_dpo_example(ex):
                valid += 1
            else:
                invalid += 1

        report = {
            "total": len(examples),
            "valid": valid,
            "invalid": invalid,
            "validity_pct": round(
                100 * valid / max(len(examples), 1),
                2,
            ),
            "first_errors": self.errors[:5],
        }

        return report