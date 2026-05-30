"""
Phase 2 unit tests for the model layer.

All tests run on CPU. The small test model (distilgpt2, ~82MB) is
downloaded once and cached by HuggingFace. No GPU required.

To avoid downloading anything during CI, set:
    export TRANSFORMERS_OFFLINE=1
and the tests that require network access will be skipped.

Run:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v -k "not slow"  # skip download tests
"""
from __future__ import annotations

import os
import json
from unittest.mock import MagicMock, patch

import pytest
import torch

# pyrefly: ignore [missing-import]
from src.models.quantization import (
    is_quantization_available,
    get_compute_dtype,
    get_bnb_config,
)
# pyrefly: ignore [missing-import]
from src.models.parameter_counter import (
    count_parameters,
    count_lora_parameters,
    lora_efficiency_report,
    print_parameter_table,
    verify_lora_initialization,
)
# pyrefly: ignore [missing-import]
from src.models.memory_estimator import (
    estimate_training_vram,
    estimate_inference_vram,
    log_current_vram,
    recommend_batch_size,
)
# pyrefly: ignore [missing-import]
from src.models.tokenizer_loader import (
    get_system_prompt,
    format_chat_prompt,
    load_tokenizer,
    SYSTEM_PROMPTS,
    _manual_chatml_format,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

_OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
_SKIP_DOWNLOAD = pytest.mark.skipif(
    _OFFLINE, reason="TRANSFORMERS_OFFLINE=1 — skipping network tests"
)

# Small model used for tests that need a real tokenizer/model
# distilgpt2: 82MB, no auth required, fast download
_TEST_MODEL = "distilgpt2"


@pytest.fixture(scope="module")
def real_tokenizer():
    """Load a small real tokenizer once for the entire test module."""
    if _OFFLINE:
        pytest.skip("Network required for tokenizer fixture")
    return load_tokenizer(_TEST_MODEL, padding_side="right", max_length=512)


@pytest.fixture(scope="module")
def small_model():
    """
    Load a tiny model (distilgpt2) for parameter counting tests.
    Cached after first download. Uses CPU only.
    """
    if _OFFLINE:
        pytest.skip("Network required for model fixture")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        _TEST_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    return model


# ─── Quantization tests ───────────────────────────────────────────────────────

class TestQuantization:
    def test_is_quantization_available_returns_bool(self):
        result = is_quantization_available()
        assert isinstance(result, bool)

    def test_quantization_unavailable_on_cpu(self):
        # On local AMD machine with no CUDA: must return False
        if not torch.cuda.is_available():
            assert is_quantization_available() is False

    def test_get_compute_dtype_bfloat16(self):
        dtype = get_compute_dtype("bfloat16")
        assert dtype == torch.bfloat16

    def test_get_compute_dtype_float16(self):
        dtype = get_compute_dtype("float16")
        assert dtype == torch.float16

    def test_get_compute_dtype_float32(self):
        dtype = get_compute_dtype("float32")
        assert dtype == torch.float32

    def test_get_compute_dtype_invalid_defaults_to_bfloat16(self):
        dtype = get_compute_dtype("invalid_dtype_xyz")
        assert dtype == torch.bfloat16

    def test_get_bnb_config_returns_none_on_cpu(self):
        if not torch.cuda.is_available():
            result = get_bnb_config()
            assert result is None

    def test_get_bnb_config_returns_config_on_gpu(self):
        if torch.cuda.is_available():
            config = get_bnb_config()
            assert config is not None
            assert config.load_in_4bit is True


# ─── Parameter counter tests ──────────────────────────────────────────────────

class TestParameterCounter:
    def test_count_parameters_structure(self, small_model):
        stats = count_parameters(small_model)
        assert "total_B" in stats
        assert "trainable_M" in stats
        assert "frozen_B" in stats
        assert "trainable_pct" in stats
        assert "by_module" in stats

    def test_total_greater_than_trainable(self, small_model):
        stats = count_parameters(small_model)
        assert stats["total_B"] * 1000 >= stats["trainable_M"]

    def test_trainable_pct_between_0_and_100(self, small_model):
        stats = count_parameters(small_model)
        assert 0.0 <= stats["trainable_pct"] <= 100.0

    def test_frozen_plus_trainable_equals_total(self, small_model):
        stats = count_parameters(small_model)
        total_M = stats["total_B"] * 1000
        # Allow small rounding error
        diff = abs(total_M - (stats["frozen_B"] * 1000 + stats["trainable_M"]))
        assert diff < 1.0  # Less than 1M difference due to rounding

    def test_by_module_non_empty(self, small_model):
        stats = count_parameters(small_model)
        assert len(stats["by_module"]) > 0

    def test_lora_efficiency_report_structure(self, small_model):
        report = lora_efficiency_report(small_model)
        assert "efficiency_ratio" in report
        assert "param_savings_pct" in report
        assert "grad_memory_saved_gb" in report

    def test_print_parameter_table_no_crash(self, small_model, capsys):
        print_parameter_table(small_model)
        captured = capsys.readouterr()
        assert "TOTAL" in captured.out
        assert "Trainable" in captured.out

    def test_lora_initialization_check_on_non_peft_model(self, small_model):
        # Non-LoRA model has no lora_B parameters → trivially True
        result = verify_lora_initialization(small_model)
        assert result is True


# ─── Memory estimator tests ───────────────────────────────────────────────────

class TestMemoryEstimator:
    def test_estimate_training_vram_structure(self):
        result = estimate_training_vram(
            params_B=7.6,
            trainable_params_M=40.0,
            batch_size=4,
            seq_len=2048,
            hidden_size=4096,
            n_layers=32,
            dtype="nf4",
        )
        required_keys = [
            "model_weights_gb", "gradients_gb", "optimizer_states_gb",
            "activations_gb", "estimated_total_gb", "with_overhead_gb",
        ]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

    def test_model_weights_nf4_approx_4gb(self):
        result = estimate_training_vram(
            params_B=7.6, trainable_params_M=40.0, dtype="nf4"
        )
        # 7.6B params × 0.5 bytes = 3.8 GB
        assert 3.0 <= result["model_weights_gb"] <= 5.0

    def test_model_weights_bf16_approx_14gb(self):
        result = estimate_training_vram(
            params_B=7.6, trainable_params_M=40.0, dtype="bfloat16"
        )
        # 7.6B params × 2 bytes = 15.2 GB
        assert 13.0 <= result["model_weights_gb"] <= 17.0

    def test_gradient_checkpointing_reduces_activations(self):
        with_ckpt = estimate_training_vram(
            params_B=7.0, trainable_params_M=40.0, gradient_checkpointing=True
        )
        without_ckpt = estimate_training_vram(
            params_B=7.0, trainable_params_M=40.0, gradient_checkpointing=False
        )
        assert with_ckpt["activations_gb"] < without_ckpt["activations_gb"]

    def test_larger_batch_increases_total(self):
        small = estimate_training_vram(params_B=7.0, trainable_params_M=40.0, batch_size=1)
        large = estimate_training_vram(params_B=7.0, trainable_params_M=40.0, batch_size=8)
        assert large["with_overhead_gb"] > small["with_overhead_gb"]

    def test_estimate_inference_vram(self):
        result = estimate_inference_vram(
            params_B=7.6, dtype="bfloat16", batch_size=1, seq_len=2048
        )
        assert result["model_weights_gb"] > 0
        assert result["kv_cache_gb"] >= 0
        assert result["total_gb"] > result["model_weights_gb"]

    def test_log_current_vram_returns_dict(self):
        result = log_current_vram()
        assert isinstance(result, dict)
        # On CPU: returns empty dict
        if not torch.cuda.is_available():
            assert result == {}

    def test_recommend_batch_size_minimum_one(self):
        # Very little available VRAM → should return 1
        batch = recommend_batch_size(
            available_vram_gb=4.0,
            model_vram_gb=3.9,
        )
        assert batch >= 1

    def test_recommend_batch_size_increases_with_vram(self):
        small = recommend_batch_size(available_vram_gb=8.0, model_vram_gb=4.0)
        large = recommend_batch_size(available_vram_gb=40.0, model_vram_gb=4.0)
        assert large >= small


# ─── Tokenizer loader tests ───────────────────────────────────────────────────

class TestTokenizerLoader:
    def test_system_prompt_lookup(self):
        prompt = get_system_prompt("instruction_following")
        assert len(prompt) > 20
        assert "helpful" in prompt.lower() or "instruction" in prompt.lower()

    def test_system_prompt_default_fallback(self):
        prompt = get_system_prompt("nonexistent_task_xyz")
        assert len(prompt) > 10

    def test_all_task_types_have_prompts(self):
        task_types = [
            "instruction_following", "structured_extraction",
            "tool_calling", "alignment_eval", "summarisation",
            "question_answering",
        ]
        for task in task_types:
            prompt = get_system_prompt(task)
            assert len(prompt) > 20, f"Prompt too short for task: {task}"

    def test_manual_chatml_format(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        result = _manual_chatml_format(messages, add_generation_prompt=True)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "You are helpful." in result

    def test_manual_chatml_no_generation_prompt(self):
        messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Hi"},
        ]
        result = _manual_chatml_format(messages, add_generation_prompt=False)
        assert "<|im_start|>assistant" not in result

    @_SKIP_DOWNLOAD
    def test_load_tokenizer_sets_pad_token(self, real_tokenizer):
        assert real_tokenizer.pad_token is not None
        assert real_tokenizer.pad_token_id is not None

    @_SKIP_DOWNLOAD
    def test_load_tokenizer_right_padding(self, real_tokenizer):
        assert real_tokenizer.padding_side == "right"

    @_SKIP_DOWNLOAD
    def test_format_chat_prompt_contains_user_message(self, real_tokenizer):
        prompt = format_chat_prompt(
            user_message="What is gradient descent?",
            task_type="instruction_following",
            tokenizer=real_tokenizer,
        )
        assert "gradient descent" in prompt.lower() or "What is gradient" in prompt

    @_SKIP_DOWNLOAD
    def test_format_chat_prompt_contains_system(self, real_tokenizer):
        system = SYSTEM_PROMPTS["instruction_following"]
        prompt = format_chat_prompt(
            user_message="Hello",
            task_type="instruction_following",
            tokenizer=real_tokenizer,
        )
        # Either the system prompt content or the role marker appears
        assert "system" in prompt.lower() or system[:20].lower() in prompt.lower()


# ─── Architecture viz tests ───────────────────────────────────────────────────

class TestArchitectureViz:
    @_SKIP_DOWNLOAD
    def test_inspect_architecture_structure(self, small_model):
        # pyrefly: ignore [missing-import]
        from src.models.architecture_viz import inspect_architecture
        info = inspect_architecture(small_model)
        required_keys = [
            "model_type", "num_layers", "hidden_size",
            "num_attention_heads", "vocab_size", "total_params_B",
        ]
        for k in required_keys:
            assert k in info, f"Missing key: {k}"

    @_SKIP_DOWNLOAD
    def test_total_params_positive(self, small_model):
        # pyrefly: ignore [missing-import]
        from src.models.architecture_viz import inspect_architecture
        info = inspect_architecture(small_model)
        assert info["total_params_B"] > 0

    @_SKIP_DOWNLOAD
    def test_get_layer_shapes_non_empty(self, small_model):
        # pyrefly: ignore [missing-import]
        from src.models.architecture_viz import get_layer_shapes
        shapes = get_layer_shapes(small_model)
        assert len(shapes) > 0
        first = shapes[0]
        assert "name" in first
        assert "shape" in first
        assert "trainable" in first

    @_SKIP_DOWNLOAD
    def test_parameter_density_sums_to_100(self, small_model):
        # pyrefly: ignore [missing-import]
        from src.models.architecture_viz import compute_parameter_density
        density = compute_parameter_density(small_model)
        total = sum(density.values())
        # Allow small floating point error
        assert abs(total - 100.0) < 1.0

    @_SKIP_DOWNLOAD
    def test_print_architecture_summary_no_crash(self, small_model, capsys):
        # pyrefly: ignore [missing-import]
        from src.models.architecture_viz import print_architecture_summary
        print_architecture_summary(small_model)
        captured = capsys.readouterr()
        assert "Architecture" in captured.out
        assert "Layers" in captured.out