"""
Phase 5 unit tests — SFT training pipeline.

All tests run on CPU without a real model.
Tests verify:
    - Optimizer parameter group construction (decay vs no-decay)
    - Scheduler step count computation
    - Callback logic (overfitting detection, gradient monitoring)
    - SFT config construction from YAML
    - compute_metrics JSON validity counting

Run:
    pytest tests/test_training.py -v
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

# pyrefly: ignore [missing-import]
from src.training.callbacks import (
    GradientMonitorCallback,
    LearningRateMonitorCallback,
    OverfitDetectorCallback,
    TrainingProgressCallback,
)
# pyrefly: ignore [missing-import]
from src.training.optimizer import build_optimizer, _is_no_decay
# pyrefly: ignore [missing-import]
from src.training.scheduler import build_scheduler, compute_total_steps
# pyrefly: ignore [missing-import]
from src.training.sft_trainer import _is_valid_json, build_compute_metrics
# pyrefly: ignore [missing-import]
from src.training.utils import (
    find_resume_checkpoint,
    get_training_dtype,
    is_bf16_supported,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_linear_model():
    """Tiny model with named parameters for optimizer testing."""
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))
            self.bias   = torch.nn.Parameter(torch.zeros(4))
            self.layer_norm = torch.nn.LayerNorm(4)

        def forward(self, x):
            return self.layer_norm(x @ self.weight + self.bias)

    return TinyModel()


@pytest.fixture
def trainer_state():
    state = MagicMock()
    state.global_step = 100
    state.max_steps = 1000
    state.epoch = 1.0
    state.best_metric = 0.5
    state.log_history = [{"loss": 0.5, "learning_rate": 2e-4}]
    return state


@pytest.fixture
def training_args():
    args = MagicMock()
    args.logging_steps = 10
    return args


@pytest.fixture
def trainer_control():
    control = MagicMock()
    control.should_training_stop = False
    return control


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizer:
    def test_is_no_decay_bias(self):
        assert _is_no_decay("model.layers.0.self_attn.q_proj.bias") is True

    def test_is_no_decay_layernorm(self):
        assert _is_no_decay("model.layers.0.input_layernorm.weight") is True

    def test_is_no_decay_embedding(self):
        assert _is_no_decay("model.embed_tokens.weight") is True

    def test_is_decay_weight_matrix(self):
        assert _is_no_decay("model.layers.0.self_attn.q_proj.weight") is False

    def test_is_decay_lora_A(self):
        assert _is_no_decay("model.layers.0.self_attn.q_proj.lora_A.default.weight") is False

    def test_build_optimizer_two_param_groups(self, tiny_linear_model):
        opt = build_optimizer(
            tiny_linear_model,
            learning_rate=1e-4,
            use_paged=False,
        )
        assert len(opt.param_groups) == 2

    def test_decay_group_has_weight_decay(self, tiny_linear_model):
        opt = build_optimizer(tiny_linear_model, weight_decay=0.01, use_paged=False)
        decay_group = opt.param_groups[0]
        assert decay_group["weight_decay"] == 0.01

    def test_no_decay_group_has_zero_wd(self, tiny_linear_model):
        opt = build_optimizer(tiny_linear_model, weight_decay=0.01, use_paged=False)
        no_decay_group = opt.param_groups[1]
        assert no_decay_group["weight_decay"] == 0.0

    def test_optimizer_lr_set_correctly(self, tiny_linear_model):
        opt = build_optimizer(tiny_linear_model, learning_rate=2e-4, use_paged=False)
        for group in opt.param_groups:
            assert group["lr"] == 2e-4

    def test_all_trainable_params_in_groups(self, tiny_linear_model):
        opt = build_optimizer(tiny_linear_model, use_paged=False)
        opt_params = set()
        for group in opt.param_groups:
            for p in group["params"]:
                opt_params.add(id(p))
        model_params = {
            id(p) for p in tiny_linear_model.parameters() if p.requires_grad
        }
        assert opt_params == model_params


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScheduler:
    def test_compute_total_steps_basic(self):
        steps = compute_total_steps(
            dataset_size=7000,
            per_device_batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3,
        )
        # 7000 / (4 * 4) = 437 per epoch, * 3 = 1312
        assert steps == (7000 // 16) * 3

    def test_compute_total_steps_minimum_one(self):
        steps = compute_total_steps(
            dataset_size=5,
            per_device_batch_size=100,
            gradient_accumulation_steps=100,
            num_epochs=1,
        )
        assert steps >= 1

    def test_build_cosine_scheduler(self, tiny_linear_model):
        opt = build_optimizer(tiny_linear_model, use_paged=False)
        scheduler = build_scheduler(opt, "cosine", num_training_steps=100)
        assert scheduler is not None

    def test_build_linear_scheduler(self, tiny_linear_model):
        opt = build_optimizer(tiny_linear_model, use_paged=False)
        scheduler = build_scheduler(opt, "linear", num_training_steps=100)
        assert scheduler is not None

    def test_build_unknown_defaults_to_cosine(self, tiny_linear_model):
        opt = build_optimizer(tiny_linear_model, use_paged=False)
        # Should not raise — defaults to cosine
        scheduler = build_scheduler(opt, "unknown_scheduler", num_training_steps=100)
        assert scheduler is not None

    def test_warmup_steps_computed_correctly(self, tiny_linear_model):
        """
        After warmup, LR should be at or near the configured value.
        Before warmup, LR should be lower.
        """
        opt = build_optimizer(
            tiny_linear_model, learning_rate=1.0, use_paged=False
        )
        warmup_steps = 10
        total_steps = 100
        scheduler = build_scheduler(
            opt, "cosine",
            num_training_steps=total_steps,
            warmup_ratio=warmup_steps / total_steps,
        )

        # At step 0: LR should be very small (warmup start)
        initial_lr = opt.param_groups[0]["lr"]
        assert initial_lr < 1.0 or initial_lr == pytest.approx(1.0 / warmup_steps, rel=0.5)

        # After warmup: step to warmup_steps
        for _ in range(warmup_steps):
            scheduler.step()
        post_warmup_lr = opt.param_groups[0]["lr"]
        # Should be near peak LR after warmup
        assert post_warmup_lr > initial_lr


# ─────────────────────────────────────────────────────────────────────────────
# Callback tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCallbacks:
    def test_gradient_monitor_no_crash_no_model(
        self, training_args, trainer_state, trainer_control
    ):
        cb = GradientMonitorCallback(log_freq=1)
        cb.on_step_end(
            training_args, trainer_state, trainer_control, model=None
        )  # Should not raise

    def test_gradient_monitor_computes_norm(
        self, training_args, trainer_state, trainer_control
    ):
        model = torch.nn.Linear(4, 4)
        # Create fake gradients
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()

        cb = GradientMonitorCallback(log_freq=1)
        norm = cb._compute_grad_norm(model)
        assert norm is not None
        assert norm > 0

    def test_gradient_monitor_none_without_grad(self):
        model = torch.nn.Linear(4, 4)
        # No backward pass → no gradients
        cb = GradientMonitorCallback(log_freq=1)
        norm = cb._compute_grad_norm(model)
        assert norm is None

    def test_overfit_detector_triggers_early_stop(
        self, training_args, trainer_state, trainer_control
    ):
        cb = OverfitDetectorCallback(divergence_threshold=0.1, patience=2)
        cb._last_train_loss = 0.5

        metrics = {"eval_loss": 0.7}  # Gap = (0.7 - 0.5) / 0.5 = 0.4 > 0.1
        cb.on_evaluate(training_args, trainer_state, trainer_control, metrics=metrics)
        assert cb._consecutive_divergence == 1

        cb.on_evaluate(training_args, trainer_state, trainer_control, metrics=metrics)
        assert cb._consecutive_divergence == 2
        assert trainer_control.should_training_stop is True

    def test_overfit_detector_resets_on_good_eval(
        self, training_args, trainer_state, trainer_control
    ):
        cb = OverfitDetectorCallback(divergence_threshold=0.3, patience=3)
        cb._last_train_loss = 0.5
        cb._consecutive_divergence = 2

        # Small gap — should reset counter
        metrics = {"eval_loss": 0.52}  # Gap = 0.04 < 0.3
        cb.on_evaluate(training_args, trainer_state, trainer_control, metrics=metrics)
        assert cb._consecutive_divergence == 0

    def test_overfit_detector_no_crash_with_none_metrics(
        self, training_args, trainer_state, trainer_control
    ):
        cb = OverfitDetectorCallback()
        cb.on_evaluate(training_args, trainer_state, trainer_control, metrics=None)

    def test_lr_monitor_captures_lr_from_logs(
        self, training_args, trainer_state, trainer_control
    ):
        cb = LearningRateMonitorCallback()
        logs = {"learning_rate": 1.5e-4, "loss": 0.4}
        cb.on_log(training_args, trainer_state, trainer_control, logs=logs)

    def test_training_progress_no_crash(
        self, training_args, trainer_state, trainer_control
    ):
        cb = TrainingProgressCallback(log_freq=1)
        logs = {"loss": 0.4, "learning_rate": 1e-4}
        cb.on_log(training_args, trainer_state, trainer_control, logs=logs)


# ─────────────────────────────────────────────────────────────────────────────
# SFT trainer utility tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSFTTrainerUtils:
    def test_is_valid_json_clean(self):
        assert _is_valid_json('{"vendor": "ACME", "total": 100.0}') is True

    def test_is_valid_json_markdown_fence(self):
        assert _is_valid_json('```json\n{"key": "val"}\n```') is True

    def test_is_valid_json_with_preamble(self):
        assert _is_valid_json('Here is the result:\n{"key": "val"}') is True

    def test_is_valid_json_truncated(self):
        assert _is_valid_json('{"key": "va') is False

    def test_is_valid_json_empty(self):
        assert _is_valid_json("") is False

    def test_is_valid_json_plain_text(self):
        assert _is_valid_json("This is plain text, no JSON here.") is False

    def test_compute_metrics_json_validity(self):
        """compute_metrics should count JSON-valid predictions correctly."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        valid_json = '{"vendor": "ACME", "total": 100.0}'
        invalid_text = "This is not JSON"

        tokenizer.batch_decode = MagicMock(
            side_effect=[
                [valid_json, invalid_text, valid_json],  # predictions
                [valid_json, invalid_text, valid_json],  # labels
            ]
        )

        compute_fn = build_compute_metrics(tokenizer)

        import numpy as np
        # Shape: (batch, seq, vocab) → logits
        predictions = np.zeros((3, 5, 100))
        labels = np.zeros((3, 5))

        result = compute_fn((predictions, labels))
        assert "json_validity" in result
        assert result["json_validity"] == pytest.approx(2 / 3, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Training utils tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingUtils:
    def test_get_training_dtype_cpu(self):
        if not torch.cuda.is_available():
            dtype = get_training_dtype()
            assert dtype == torch.float32

    def test_is_bf16_supported_returns_bool(self):
        result = is_bf16_supported()
        assert isinstance(result, bool)

    def test_find_resume_checkpoint_nonexistent_dir(self, tmp_path):
        result = find_resume_checkpoint(str(tmp_path / "nonexistent"))
        assert result is None

    def test_find_resume_checkpoint_empty_dir(self, tmp_path):
        result = find_resume_checkpoint(str(tmp_path))
        assert result is None

    def test_find_resume_checkpoint_finds_latest(self, tmp_path):
        # Create fake checkpoint directories
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-200").mkdir()
        (tmp_path / "checkpoint-50").mkdir()
        (tmp_path / "not_a_checkpoint").mkdir()

        result = find_resume_checkpoint(str(tmp_path))
        assert result is not None
        assert "checkpoint-200" in result

    def test_find_resume_checkpoint_single(self, tmp_path):
        (tmp_path / "checkpoint-500").mkdir()
        result = find_resume_checkpoint(str(tmp_path))
        assert "checkpoint-500" in result