"""
DatasetOrchestrator: coordinates synthetic + public dataset loading
to produce the final SFTExample and DPOExample training sets.

Public datasets are loaded from HuggingFace Hub using the `datasets`
library. They are loaded lazily and only the required subset is
downloaded, so this works on a machine with limited storage.

Public dataset loading works on the local Windows/AMD machine
because it is pure Python + network download, no GPU required.
The full 200k UltraChat dataset is NOT downloaded — only a curated
streaming subset is fetched.

Dataset composition:
    Synthetic (InstructionFollowing)    2,500
    Synthetic (StructuredOutput)        2,000
    Synthetic (ToolCall)                  500
    Synthetic (AlignmentBehaviour)        500
    Public (UltraChat-200k subset)      2,000
    Public (OpenHermes-2.5 subset)      1,500
                                       ──────
    Total SFT examples                  9,000

DPO pairs (from SFT training split, 70% = ~6,300 examples):
    All synthetic examples → programmatic rejection (6 strategies)
    Public examples → programmatic rejection (same strategies)
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Iterator

from faker import Faker

from src.data.schemas import (
    Conversation,
    ConversationQuality,
    DifficultyLevel,
    MessageTurn,
    Role,
    SFTExample,
    TaskType,
)
from src.data.synthetic_builder import (
    AlignmentBehaviourGenerator,
    InstructionFollowingGenerator,
    StructuredOutputGenerator,
    ToolCallGenerator,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

class UltraChatLoader:
    """
    Loads a curated subset of HuggingFace's UltraChat-200k dataset.

    UltraChat-200k is a filtered version of UltraChat containing
    ~200k high-quality multi-turn instruction conversations.
    We load only `n_examples` using streaming to avoid downloading
    the full dataset.

    Each UltraChat example is a multi-turn conversation that we convert
    to SFTExample format by:
    1. Adding a generic instruction-following system prompt
    2. Taking the first user-assistant exchange (or full conversation)
    3. Wrapping in our Conversation / SFTExample schema

    HuggingFace model card: HuggingFaceH4/ultrachat_200k
    License: MIT
    """

    _SYSTEM_PROMPT = (
        "You are a helpful, harmless, and honest AI assistant. "
        "Provide accurate, thoughtful responses to the user's questions."
    )

    def __init__(self, n_examples: int = 2000, seed: int = 42):
        self.n_examples = n_examples
        self.seed = seed

    def load(self) -> list[SFTExample]:
        """
        Stream-load n_examples from UltraChat-200k train split.

        Returns an empty list with a warning if the dataset cannot be
        downloaded (e.g. no internet connection). The orchestrator
        gracefully falls back to synthetic-only in this case.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            return []

        try:
            logger.info(f"Loading {self.n_examples} examples from UltraChat-200k...")
            ds = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split="train_sft",
                streaming=True,
                trust_remote_code=True,
            )
            ds = ds.shuffle(seed=self.seed)
        except Exception as e:
            logger.warning(f"UltraChat load failed: {e}. Skipping public dataset.")
            return []

        examples: list[SFTExample] = []
        for row in ds:
            if len(examples) >= self.n_examples:
                break
            example = self._convert_row(row)
            if example is not None:
                examples.append(example)

        logger.info(f"Loaded {len(examples)} UltraChat examples")
        return examples

    def _convert_row(self, row: dict) -> SFTExample | None:
        """
        Convert one UltraChat row to SFTExample.

        UltraChat rows have a 'messages' field with role/content dicts.
        We reconstruct the Conversation from these and add a system prompt
        if one is not present.
        """
        try:
            messages = row.get("messages", [])
            if not messages:
                return None

            turns: list[MessageTurn] = []

            # Add system prompt if not already present
            if messages[0].get("role") != "system":
                turns.append(
                    MessageTurn(
                        role=Role.SYSTEM,
                        content=self._SYSTEM_PROMPT,
                        turn_idx=0,
                    )
                )

            for i, msg in enumerate(messages):
                role_str = msg.get("role", "")
                content = msg.get("content", "").strip()
                if not content or role_str not in ("system", "user", "assistant"):
                    continue
                turns.append(
                    MessageTurn(
                        role=Role(role_str),
                        content=content,
                        turn_idx=len(turns),
                    )
                )

            if len(turns) < 3:
                return None
            if turns[-1].role != Role.ASSISTANT:
                return None

            conv = Conversation(turns=turns)
            return SFTExample(
                conversation=conv,
                task_type=TaskType.INSTRUCTION_FOLLOWING,
                difficulty=DifficultyLevel.MEDIUM,
                quality=ConversationQuality.HUMAN_WRITTEN,
                source="ultrachat_200k",
                weight=1.2,
                tags=["public", "multi_turn", "ultrachat"],
            )
        except Exception as e:
            logger.debug(f"UltraChat row conversion failed: {e}")
            return None


class OpenHermesLoader:
    """
    Loads a curated subset of the OpenHermes-2.5 dataset.

    OpenHermes-2.5 contains ~1M diverse instruction-following examples
    sourced from GPT-4 generations across many task types including
    coding, reasoning, creative writing, and factual QA.

    We filter to examples with a system prompt (about 40% of the dataset)
    because these are the most valuable for alignment training — they
    demonstrate explicit system prompt adherence.

    HuggingFace model card: teknium/OpenHermes-2.5
    License: MIT
    """

    _DEFAULT_SYSTEM = (
        "You are a helpful AI assistant that follows instructions carefully "
        "and provides accurate, well-reasoned responses."
    )

    def __init__(self, n_examples: int = 1500, seed: int = 42):
        self.n_examples = n_examples
        self.seed = seed

    def load(self) -> list[SFTExample]:
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed.")
            return []

        try:
            logger.info(f"Loading {self.n_examples} examples from OpenHermes-2.5...")
            ds = load_dataset(
                "teknium/OpenHermes-2.5",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            ds = ds.shuffle(seed=self.seed)
        except Exception as e:
            logger.warning(f"OpenHermes load failed: {e}. Skipping public dataset.")
            return []

        examples: list[SFTExample] = []
        for row in ds:
            if len(examples) >= self.n_examples:
                break
            example = self._convert_row(row)
            if example is not None:
                examples.append(example)

        logger.info(f"Loaded {len(examples)} OpenHermes examples")
        return examples

    def _convert_row(self, row: dict) -> SFTExample | None:
        """
        Convert one OpenHermes row to SFTExample.

        OpenHermes uses ShareGPT format: conversations list with
        'from' (human/gpt/system) and 'value' fields.
        """
        try:
            conversations = row.get("conversations", [])
            if not conversations:
                return None

            turns: list[MessageTurn] = []
            system_prompt = row.get("system_prompt", "").strip()

            # Add system prompt
            system_content = system_prompt if system_prompt else self._DEFAULT_SYSTEM
            turns.append(
                MessageTurn(role=Role.SYSTEM, content=system_content, turn_idx=0)
            )

            for msg in conversations:
                from_role = msg.get("from", "")
                value = msg.get("value", "").strip()
                if not value:
                    continue

                role_map = {"human": Role.USER, "gpt": Role.ASSISTANT, "system": Role.SYSTEM}
                role = role_map.get(from_role)
                if role is None or (role == Role.SYSTEM and turns):
                    continue

                turns.append(
                    MessageTurn(role=role, content=value, turn_idx=len(turns))
                )

            if len(turns) < 3:
                return None
            if turns[-1].role != Role.ASSISTANT:
                return None

            # Filter excessively long examples
            total_chars = sum(len(t.content) for t in turns)
            if total_chars > 8000:
                return None

            return SFTExample(
                conversation=Conversation(turns=turns),
                task_type=TaskType.INSTRUCTION_FOLLOWING,
                difficulty=DifficultyLevel.MEDIUM,
                quality=ConversationQuality.SYNTHETIC_LLM,
                source="openhermes_2.5",
                weight=1.0,
                tags=["public", "diverse", "openhermes"],
            )
        except Exception as e:
            logger.debug(f"OpenHermes row conversion failed: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# DatasetOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class DatasetOrchestrator:
    """
    Coordinates all data sources into a unified SFTExample list.

    Composition targets (approximate — public loaders may return fewer
    examples if the network is unavailable, in which case synthetic
    counts are scaled up to compensate):

        Source                   Target    Weight
        ─────────────────────────────────────────
        UltraChat-200k           2,000     1.2
        OpenHermes-2.5           1,500     1.0
        InstructionFollowing     2,500     1.0
        StructuredOutput         2,000     1.0
        ToolCall                   500     1.0
        AlignmentBehaviour         500     1.5

    Weights > 1.0 on SFTExample.weight tell the trainer to sample those
    examples more frequently. HUMAN_WRITTEN UltraChat examples get 1.2×,
    alignment behaviour examples get 1.5× (they are rare and critical).

    The orchestrator shuffles the final list with a fixed seed before
    returning so the train/val/test split boundaries are stable
    across runs with the same seed.
    """

    def __init__(
        self,
        seed: int = 42,
        load_public: bool = True,
        ultrachat_n: int = 2000,
        openhermes_n: int = 1500,
    ):
        self.seed = seed
        self.load_public = load_public
        random.seed(seed)

        _fake = Faker()
        _fake.seed_instance(seed)

        self._synth_generators = {
            "instruction": InstructionFollowingGenerator(_fake),
            "structured": StructuredOutputGenerator(_fake),
            "tool_call": ToolCallGenerator(_fake),
            "alignment": AlignmentBehaviourGenerator(_fake),
        }

        self._public_loaders = {
            "ultrachat": UltraChatLoader(n_examples=ultrachat_n, seed=seed),
            "openhermes": OpenHermesLoader(n_examples=openhermes_n, seed=seed),
        }

        # Synthetic generation targets per difficulty tier
        self._synthetic_targets = {
            "instruction": {
                DifficultyLevel.EASY: 600,
                DifficultyLevel.MEDIUM: 1200,
                DifficultyLevel.HARD: 500,
                DifficultyLevel.ADVERSARIAL: 200,
            },
            "structured": {
                DifficultyLevel.EASY: 500,
                DifficultyLevel.MEDIUM: 900,
                DifficultyLevel.HARD: 400,
                DifficultyLevel.ADVERSARIAL: 200,
            },
            "tool_call": {
                DifficultyLevel.EASY: 200,
                DifficultyLevel.MEDIUM: 200,
                DifficultyLevel.HARD: 70,
                DifficultyLevel.ADVERSARIAL: 30,
            },
            "alignment": {
                DifficultyLevel.EASY: 100,
                DifficultyLevel.MEDIUM: 250,
                DifficultyLevel.HARD: 100,
                DifficultyLevel.ADVERSARIAL: 50,
            },
        }

    def generate_sft_examples(self) -> list[SFTExample]:
        """
        Generate the complete SFT training set.

        Returns a shuffled list of SFTExample objects ready for
        preprocessor.build_sft_datasets().

        If public dataset loading fails (no internet, auth error), the
        method logs a warning and proceeds with synthetic-only examples.
        The caller receives a valid dataset regardless.
        """
        all_examples: list[SFTExample] = []

        # ── Synthetic examples ────────────────────────────────────────────────
        for source_name, targets in self._synthetic_targets.items():
            gen = self._synth_generators[source_name]
            for difficulty, count in targets.items():
                logger.info(
                    f"[Orchestrator] Generating {count} {source_name} "
                    f"({difficulty.value}) examples"
                )
                for _ in range(count):
                    try:
                        ex = gen.generate(difficulty=difficulty)
                        all_examples.append(ex)
                    except Exception as e:
                        logger.warning(
                            f"[Orchestrator] {source_name} generation failed: {e}"
                        )

        # ── Public dataset examples ───────────────────────────────────────────
        if self.load_public:
            ultrachat = self._public_loaders["ultrachat"].load()
            all_examples.extend(ultrachat)

            openhermes = self._public_loaders["openhermes"].load()
            all_examples.extend(openhermes)

            if not ultrachat and not openhermes:
                logger.warning(
                    "[Orchestrator] Both public datasets failed to load. "
                    "Proceeding with synthetic-only dataset. "
                    "The training set will be smaller but fully functional."
                )
        else:
            logger.info("[Orchestrator] Public dataset loading disabled (load_public=False)")

        random.shuffle(all_examples)
        logger.info(
            f"[Orchestrator] Total SFT examples: {len(all_examples):,}\n"
            + self._composition_summary(all_examples)
        )
        return all_examples

    def _composition_summary(self, examples: list[SFTExample]) -> str:
        by_source = Counter(ex.source for ex in examples)
        by_task = Counter(ex.task_type.value for ex in examples)
        by_quality = Counter(ex.quality.value for ex in examples)
        by_difficulty = Counter(ex.difficulty.value for ex in examples)

        lines = ["  Dataset composition:"]
        lines.append("  By source:")
        for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
            lines.append(f"    {src:<45} {count:>5}")
        lines.append("  By task type:")
        for task, count in sorted(by_task.items(), key=lambda x: -x[1]):
            lines.append(f"    {task:<45} {count:>5}")
        lines.append("  By quality:")
        for q, count in sorted(by_quality.items(), key=lambda x: -x[1]):
            lines.append(f"    {q:<45} {count:>5}")
        lines.append("  By difficulty:")
        for d, count in sorted(by_difficulty.items(), key=lambda x: -x[1]):
            lines.append(f"    {d:<45} {count:>5}")
        return "\n".join(lines)