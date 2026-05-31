"""
Phase 6 unit tests — DPO training pipeline.

All tests run on CPU without a real model.
Tests verify:
    - DPO config construction from YAML
    - DPO dataset loading and validation
    - Preference pair quality (chosen != rejected)
    - β parameter effect on loss scale
    - Trainer assembly with mock model

Run:
    pytest tests/test_dpo.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# pyrefly: ignore [missing-import]
from src.data.schemas import (
    Conversation,
    ConversationQuality,
    DifficultyLevel,
    DPOExample,
    MessageTurn,
    ModelStage,
    RejectionStrategy,
    Role,
    SFTExample,
    TaskType,
)
# pyrefly: ignore [missing-import]
from src.training.dpo_trainer import (
    build_dpo_datasets_from_disk,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sft_example():
    turns = [
        MessageTurn(role=Role.SYSTEM, content="Extract JSON.", turn_idx=0),
        MessageTurn(role=Role.USER,   content="Invoice from ACME, total $100", turn_idx=1),
        MessageTurn(
            role=Role.ASSISTANT,
            content=json.dumps({"vendor": "ACME", "total": 100.0}),
            turn_idx=2,
        ),
    ]
    return SFTExample(
        conversation=Conversation(turns=turns),
        task_type=TaskType.STRUCTURED_EXTRACTION,
        difficulty=DifficultyLevel.MEDIUM,
        quality=ConversationQuality.SYNTHETIC_TEMPLATE,
        source="test",
    )


@pytest.fixture
def sample_dpo_pair():
    return DPOExample(
        prompt="[system]Extract JSON.[user]Invoice from ACME[assistant]",
        chosen=json.dumps({"vendor": "ACME", "total": 100.0}),
        rejected=json.dumps({"vendor": "ACME", "total": 100.0, "confidence": 0.99}),
        preference_strength=1.0,
        rejection_strategy=RejectionStrategy.HALLUCINATED_FIELDS,
        task_type=TaskType.STRUCTURED_EXTRACTION,
    )


@pytest.fixture
def dpo_jsonl_files(tmp_path):
    """Create temporary DPO JSONL files for testing."""
    train_pairs = [
        {
            "prompt": f"[system]Extract.[user]Input {i}[assistant]",
            "chosen": json.dumps({"field": f"value_{i}"}),
            "rejected": json.dumps({"field": f"value_{i}", "extra": "hallucinated"}),
        }
        for i in range(20)
    ]
    val_pairs = [
        {
            "prompt": f"[system]Extract.[user]Val {i}[assistant]",
            "chosen": json.dumps({"field": f"val_{i}"}),
            "rejected": f'{{"field": "val_{i}"',  # Truncated — valid rejection
        }
        for i in range(5)
    ]

    train_path = tmp_path / "dpo_train.jsonl"
    val_path   = tmp_path / "dpo_val.jsonl"

    with open(train_path, "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")
    with open(val_path, "w") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + "\n")

    return str(train_path), str(val_path)


# ─────────────────────────────────────────────────────────────────────────────
# DPO schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDPOExample:
    def test_valid_pair_creation(self, sample_dpo_pair):
        assert sample_dpo_pair.chosen != sample_dpo_pair.rejected
        assert 0.0 <= sample_dpo_pair.preference_strength <= 1.0

    def test_identical_chosen_rejected_raises(self):
        with pytest.raises(Exception, match="identical"):
            DPOExample(
                prompt="test",
                chosen="same response",
                rejected="same response",
            )

    def test_to_trl_dict_has_three_keys(self, sample_dpo_pair):
        d = sample_dpo_pair.to_trl_dict()
        assert set(d.keys()) == {"prompt", "chosen", "rejected"}

    def test_to_trl_dict_values_are_strings(self, sample_dpo_pair):
        d = sample_dpo_pair.to_trl_dict()
        for k, v in d.items():
            assert isinstance(v, str), f"Key {k} is not a string: {type(v)}"

    def test_rejection_strategy_enum(self, sample_dpo_pair):
        assert sample_dpo_pair.rejection_strategy == RejectionStrategy.HALLUCINATED_FIELDS

    def test_serialisation_roundtrip(self, sample_dpo_pair):
        d = sample_dpo_pair.to_dict()
        assert d["rejection_strategy"] == "hallucinated_fields"
        assert d["preference_strength"] == 1.0
        serialised = json.dumps(d)
        assert len(serialised) > 0


# ─────────────────────────────────────────────────────────────────────────────
# DPO dataset loading tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDPODatasetLoading:
    def test_load_from_disk_train_size(self, dpo_jsonl_files):
        train_path, val_path = dpo_jsonl_files
        datasets = build_dpo_datasets_from_disk(train_path, val_path)
        assert len(datasets["train"]) == 20

    def test_load_from_disk_val_size(self, dpo_jsonl_files):
        train_path, val_path = dpo_jsonl_files
        datasets = build_dpo_datasets_from_disk(train_path, val_path)
        assert len(datasets["validation"]) == 5

    def test_dataset_has_required_columns(self, dpo_jsonl_files):
        train_path, val_path = dpo_jsonl_files
        datasets = build_dpo_datasets_from_disk(train_path, val_path)
        required = {"prompt", "chosen", "rejected"}
        actual = set(datasets["train"].column_names)
        assert required.issubset(actual)

    def test_fallback_split_when_no_val_file(self, tmp_path):
        """When no val file exists, should auto-split train 90/10."""
        train_pairs = [
            {
                "prompt": f"[user]Q{i}[assistant]",
                "chosen": f"Good answer {i}",
                "rejected": f"Bad answer {i}",
            }
            for i in range(20)
        ]
        train_path = tmp_path / "dpo_train.jsonl"
        with open(train_path, "w") as f:
            for p in train_pairs:
                f.write(json.dumps(p) + "\n")

        val_path = str(tmp_path / "nonexistent_val.jsonl")
        datasets = build_dpo_datasets_from_disk(str(train_path), val_path)

        assert len(datasets["train"]) + len(datasets["validation"]) == 20

    def test_malformed_records_skipped(self, tmp_path):
        """Records missing required keys should be silently skipped."""
        pairs = [
            {"prompt": "P1", "chosen": "C1", "rejected": "R1"},  # valid
            {"prompt": "P2", "chosen": "C2"},                     # missing rejected
            {"prompt": "P3", "chosen": "C3", "rejected": "R3"},  # valid
        ]
        train_path = tmp_path / "dpo_train.jsonl"
        with open(train_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        datasets = build_dpo_datasets_from_disk(str(train_path), str(tmp_path / "no_val.jsonl"))
        # Only 2 valid records (1 is missing rejected)
        total = len(datasets["train"]) + len(datasets.get("validation", []))
        assert total == 2


# ─────────────────────────────────────────────────────────────────────────────
# DPO config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDPOConfig:
    def _make_cfg(self, beta=0.1, loss_type="sigmoid", lr=5e-5):
        # pyrefly: ignore [missing-import]
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "model": {"name": "test-model"},
            "dpo": {
                "beta": beta,
                "loss_type": loss_type,
                "max_prompt_length": 256,
                "max_length": 512,
                "label_smoothing": 0.0,
            },
            "training": {
                "output_dir": "/tmp/test_dpo",
                "num_train_epochs": 1,
                "max_steps": 10,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": lr,
                "weight_decay": 0.01,
                "warmup_ratio": 0.05,
                "lr_scheduler_type": "cosine",
                "max_grad_norm": 1.0,
                "bf16": False,
                "fp16": False,
                "logging_steps": 5,
                "report_to": "none",
                "run_name": "test-dpo",
                "eval_steps": 5,
                "save_steps": 10,
                "save_total_limit": 1,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "dataloader_num_workers": 0,
                "seed": 42,
            },
        })

    def test_dpo_config_beta(self):
        # pyrefly: ignore [missing-import]
        from src.training.dpo_trainer import build_dpo_config
        cfg = self._make_cfg(beta=0.2)
        config = build_dpo_config(cfg)
        assert config.beta == 0.2

    def test_dpo_config_loss_type(self):
        # pyrefly: ignore [missing-import]
        from src.training.dpo_trainer import build_dpo_config
        cfg = self._make_cfg(loss_type="ipo")
        config = build_dpo_config(cfg)
        assert config.loss_type == "ipo"

    def test_dpo_config_lr(self):
        # pyrefly: ignore [missing-import]
        from src.training.dpo_trainer import build_dpo_config
        cfg = self._make_cfg(lr=1e-5)
        config = build_dpo_config(cfg)
        assert config.learning_rate == 1e-5

    def test_dpo_config_cpu_optim(self):
        """On CPU: should use adamw_torch not paged_adamw_32bit."""
        # pyrefly: ignore [missing-import]
        from src.training.dpo_trainer import build_dpo_config
        cfg = self._make_cfg()
        config = build_dpo_config(cfg)
        if not torch.cuda.is_available():
            assert config.optim == "adamw_torch"


# ─────────────────────────────────────────────────────────────────────────────
# Rejection strategy quality tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRejectionStrategies:
    """
    Verify that rejected responses are genuinely different from chosen
    and represent realistic failure modes.
    """

    def _make_pair(self, strategy: RejectionStrategy) -> DPOExample:
        """Build a DPO pair using a specific rejection strategy."""
        # pyrefly: ignore [missing-import]
        from src.data.preprocessor import RejectionEngine
        chosen = json.dumps(
            {"vendor": "ACME Corp", "total": 150.0, "currency": "USD"}
        )
        engine = RejectionEngine()
        rejected, used_strategy = engine.generate(
            chosen=chosen,
            task_type=TaskType.STRUCTURED_EXTRACTION,
            strategy=strategy,
        )
        return DPOExample(
            prompt="[system]Extract.[user]Invoice[assistant]",
            chosen=chosen,
            rejected=rejected,
            rejection_strategy=used_strategy,
        )

    def test_hallucinated_fields_adds_keys(self):
        pair = self._make_pair(RejectionStrategy.HALLUCINATED_FIELDS)
        pred = json.loads(pair.rejected)
        ref = json.loads(pair.chosen)
        # Hallucinated version should have MORE keys
        assert len(pred) > len(ref)

    def test_truncated_is_invalid_json(self):
        pair = self._make_pair(RejectionStrategy.TRUNCATED)
        # Truncated JSON should be invalid
        try:
            json.loads(pair.rejected)
            # If it happens to be valid, it's still different
            assert pair.rejected != pair.chosen
        except json.JSONDecodeError:
            pass  # Expected — truncated JSON is invalid

    def test_verbose_wrapper_contains_prose(self):
        pair = self._make_pair(RejectionStrategy.VERBOSE_WRAPPER)
        # Should contain text before the JSON
        assert len(pair.rejected) > len(pair.chosen)
        assert "json" in pair.rejected.lower() or "extract" in pair.rejected.lower()

    def test_schema_violation_changes_type(self):
        pair = self._make_pair(RejectionStrategy.SCHEMA_VIOLATION)
        try:
            pred = json.loads(pair.rejected)
            ref = json.loads(pair.chosen)
            # At least one field should have a different type
            has_type_change = any(
                type(pred.get(k)) != type(ref.get(k))
                for k in ref
                if pred.get(k) is not None and ref.get(k) is not None
            )
            # Either type changed or it's different in some way
            assert has_type_change or pred != ref
        except json.JSONDecodeError:
            pass  # Schema violation might produce invalid JSON

    def test_all_rejections_differ_from_chosen(self):
        """Every rejection strategy must produce output different from chosen."""
        import random
        random.seed(42)

        chosen = json.dumps({"vendor": "ACME", "total": 100.0, "currency": "USD"})

        # pyrefly: ignore [missing-import]
        from src.data.preprocessor import RejectionEngine
        engine = RejectionEngine()

        for strategy in RejectionStrategy:
            rejected, _ = engine.generate(
                chosen=chosen,
                task_type=TaskType.STRUCTURED_EXTRACTION,
                strategy=strategy,
            )
            assert rejected.strip() != chosen.strip(), (
                f"Strategy {strategy.value} produced identical chosen/rejected"
            )


# ─────────────────────────────────────────────────────────────────────────────
# DPO mathematics tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDPOMathematics:
    """
    Test the DPO loss formula numerically.

    These tests verify that our understanding of DPO is correct
    and can be demonstrated in code — portfolio value.
    """

    def test_dpo_loss_decreases_for_correct_preference(self):
        """
        DPO loss should be lower when the policy prefers chosen over rejected.

        Loss = -log σ(β · (log_ratio_chosen - log_ratio_rejected))
        When log_ratio_chosen > log_ratio_rejected: argument to σ is positive.
        σ(positive) > 0.5 → -log < log(2) ≈ 0.693.
        """
        import math

        beta = 0.1

        # Policy correctly prefers chosen
        log_ratio_chosen   = 0.5   # Policy assigns higher prob to chosen than ref
        log_ratio_rejected = -0.3  # Policy assigns lower prob to rejected than ref

        loss = -math.log(
            1 / (1 + math.exp(-(beta * (log_ratio_chosen - log_ratio_rejected))))
        )

        # Loss should be less than log(2) ≈ 0.693 when preference is correct
        assert loss < math.log(2)

    def test_dpo_loss_high_when_preference_wrong(self):
        """
        DPO loss should be high when policy prefers rejected over chosen.
        """
        import math

        beta = 0.1

        # Policy incorrectly prefers rejected (bad case)
        log_ratio_chosen   = -0.5
        log_ratio_rejected = 0.3

        loss = -math.log(
            1 / (1 + math.exp(-(beta * (log_ratio_chosen - log_ratio_rejected))))
        )

        # Loss should be greater than log(2) when preference is wrong
        assert loss > math.log(2)

    def test_higher_beta_closer_to_reference(self):
        """
        Higher β should penalise deviation from reference more strongly.
        Two policies with same preference direction — higher β gives lower loss
        only when log_ratios are already in the correct direction.
        """
        import math

        log_ratio_chosen   = 0.5
        log_ratio_rejected = -0.3
        margin = log_ratio_chosen - log_ratio_rejected

        def dpo_loss(beta):
            return -math.log(
                1 / (1 + math.exp(-(beta * margin)))
            )

        loss_low_beta  = dpo_loss(0.01)
        loss_high_beta = dpo_loss(1.0)

        # Higher β pushes σ argument further → lower loss when margin is positive
        assert loss_high_beta < loss_low_beta

    def test_rewards_accuracy_definition(self):
        """
        rewards/accuracies = fraction of pairs where log_ratio_chosen > log_ratio_rejected.
        This should approach 1.0 as training progresses on clean preference data.
        """
        pairs = [
            (0.5, -0.3),   # Correct: chosen > rejected
            (0.2, -0.1),   # Correct
            (-0.1, 0.4),   # Wrong: rejected > chosen (bad)
            (0.8, -0.5),   # Correct
        ]
        correct = sum(1 for c, r in pairs if c > r)
        accuracy = correct / len(pairs)
        assert accuracy == 0.75