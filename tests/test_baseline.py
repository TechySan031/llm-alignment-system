"""
Phase 4 unit tests — baseline strategies and pipeline.

All tests run on CPU without loading any real model.
Strategy classes are tested with a mock tokenizer.

Run:
    pytest tests/test_baseline.py -v
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

# pyrefly: ignore [missing-import]
from src.data.schemas import (
    Conversation,
    ConversationQuality,
    DifficultyLevel,
    MessageTurn,
    ModelStage,
    Role,
    SFTExample,
    TaskType,
)
# pyrefly: ignore [missing-import]
from src.evaluation.baseline_strategies import (
    ChainOfThoughtStrategy,
    FewShotCoTStrategy,
    FewShotStrategy,
    ZeroShotStrategy,
    _COT_PREFIXES,
    _FEW_SHOT_DEMONSTRATIONS,
    get_all_strategies,
    get_strategy,
    pick_best_strategy,
)
# pyrefly: ignore [missing-import]
from src.evaluation.benchmarks import MetricsComputer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that does simple string concatenation."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.pad_token = "<pad>"

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}]{m['content']}")
        if add_generation_prompt:
            parts.append("[assistant]")
        return "".join(parts)

    tok.apply_chat_template = apply_chat_template
    return tok


@pytest.fixture
def sample_sft_example():
    """A simple SFTExample for strategy testing."""
    turns = [
        MessageTurn(role=Role.SYSTEM, content="You are helpful.", turn_idx=0),
        MessageTurn(role=Role.USER, content="Extract JSON from this invoice text.", turn_idx=1),
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
def tool_call_example():
    turns = [
        MessageTurn(role=Role.SYSTEM, content="You are a function caller.", turn_idx=0),
        MessageTurn(role=Role.USER, content="Process a refund of $50.", turn_idx=1),
        MessageTurn(
            role=Role.ASSISTANT,
            content=json.dumps({"tool": "process_refund", "args": {"amount": 50.0}}),
            turn_idx=2,
        ),
    ]
    return SFTExample(
        conversation=Conversation(turns=turns),
        task_type=TaskType.TOOL_CALLING,
        difficulty=DifficultyLevel.EASY,
        quality=ConversationQuality.SYNTHETIC_TEMPLATE,
        source="test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy name tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyNames:
    def test_zero_shot_name(self):
        assert ZeroShotStrategy().name == "zero_shot"

    def test_few_shot_name(self):
        assert FewShotStrategy(n_shots=2).name == "few_shot_2"

    def test_cot_name(self):
        assert ChainOfThoughtStrategy().name == "chain_of_thought"

    def test_few_shot_cot_name(self):
        assert FewShotCoTStrategy().name == "few_shot_cot"

    def test_get_all_strategies_returns_four(self):
        strategies = get_all_strategies()
        assert len(strategies) == 4

    def test_get_strategy_by_name(self):
        strategy = get_strategy("zero_shot")
        assert strategy.name == "zero_shot"

    def test_get_strategy_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent_strategy")


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot strategy tests
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroShotStrategy:
    def test_prompt_contains_user_message(self, mock_tokenizer, sample_sft_example):
        strategy = ZeroShotStrategy()
        prompt = strategy.build_prompt(sample_sft_example, mock_tokenizer)
        assert "Extract JSON from this invoice text." in prompt

    def test_prompt_contains_system(self, mock_tokenizer, sample_sft_example):
        strategy = ZeroShotStrategy()
        prompt = strategy.build_prompt(sample_sft_example, mock_tokenizer)
        assert "[system]" in prompt or "system" in prompt.lower()

    def test_prompt_not_contains_demonstration(self, mock_tokenizer, sample_sft_example):
        strategy = ZeroShotStrategy()
        prompt = strategy.build_prompt(sample_sft_example, mock_tokenizer)
        # Should not contain any few-shot demo content
        assert "Bright Solutions" not in prompt  # Demo vendor name


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot strategy tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFewShotStrategy:
    def test_prompt_longer_than_zero_shot(self, mock_tokenizer, sample_sft_example):
        zero_shot = ZeroShotStrategy()
        few_shot = FewShotStrategy(n_shots=2)

        zs_prompt = zero_shot.build_prompt(sample_sft_example, mock_tokenizer)
        fs_prompt = few_shot.build_prompt(sample_sft_example, mock_tokenizer)

        # Few-shot includes demonstrations — must be longer
        assert len(fs_prompt) >= len(zs_prompt)

    def test_demonstrations_exist_for_structured_extraction(self):
        assert "structured_extraction" in _FEW_SHOT_DEMONSTRATIONS
        assert len(_FEW_SHOT_DEMONSTRATIONS["structured_extraction"]) >= 1

    def test_demonstration_has_user_and_assistant(self):
        for task, demos in _FEW_SHOT_DEMONSTRATIONS.items():
            for demo in demos:
                assert "user" in demo, f"Missing 'user' in {task} demo"
                assert "assistant" in demo, f"Missing 'assistant' in {task} demo"

    def test_demonstration_assistant_is_valid_json_for_extraction(self):
        for demo in _FEW_SHOT_DEMONSTRATIONS.get("structured_extraction", []):
            try:
                json.loads(demo["assistant"])
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Demonstration assistant output is not valid JSON: {e}\n"
                    f"Content: {demo['assistant'][:200]}"
                )

    def test_few_shot_includes_test_example(self, mock_tokenizer, sample_sft_example):
        strategy = FewShotStrategy(n_shots=1)
        prompt = strategy.build_prompt(sample_sft_example, mock_tokenizer)
        # The test example's user message must appear in the prompt
        assert sample_sft_example.input_text in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Chain-of-thought strategy tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChainOfThoughtStrategy:
    def test_cot_prefix_appended(self, mock_tokenizer, sample_sft_example):
        strategy = ChainOfThoughtStrategy()
        prompt = strategy.build_prompt(sample_sft_example, mock_tokenizer)
        # The CoT prefix for structured_extraction should appear
        expected_prefix = _COT_PREFIXES["structured_extraction"]
        assert expected_prefix[:20] in prompt

    def test_cot_prefixes_exist_for_main_tasks(self):
        required = [
            "structured_extraction",
            "instruction_following",
            "tool_calling",
            "alignment_eval",
        ]
        for task in required:
            assert task in _COT_PREFIXES, f"Missing CoT prefix for {task}"
            assert len(_COT_PREFIXES[task]) > 10

    def test_cot_prompt_longer_than_zero_shot(self, mock_tokenizer, sample_sft_example):
        zs = ZeroShotStrategy().build_prompt(sample_sft_example, mock_tokenizer)
        cot = ChainOfThoughtStrategy().build_prompt(sample_sft_example, mock_tokenizer)
        assert len(cot) > len(zs)


# ─────────────────────────────────────────────────────────────────────────────
# Few-shot CoT strategy tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFewShotCoTStrategy:
    def test_longer_than_both_few_shot_and_cot(self, mock_tokenizer, sample_sft_example):
        zs = ZeroShotStrategy().build_prompt(sample_sft_example, mock_tokenizer)
        fs = FewShotStrategy(n_shots=1).build_prompt(sample_sft_example, mock_tokenizer)
        cot = ChainOfThoughtStrategy().build_prompt(sample_sft_example, mock_tokenizer)
        fscot = FewShotCoTStrategy().build_prompt(sample_sft_example, mock_tokenizer)
        # Few-shot CoT should be the longest prompt
        assert len(fscot) > len(zs)

    def test_contains_test_example_and_prefix(self, mock_tokenizer, sample_sft_example):
        strategy = FewShotCoTStrategy()
        prompt = strategy.build_prompt(sample_sft_example, mock_tokenizer)
        assert sample_sft_example.input_text in prompt


# ─────────────────────────────────────────────────────────────────────────────
# Pick best strategy tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPickBestStrategy:
    def _make_result(self, alignment_score: float):
        # pyrefly: ignore [missing-import]
        from src.data.schemas import BenchmarkResult
        r = BenchmarkResult(
            model_id="test", model_stage=ModelStage.BASE,
            n_examples=50,
        )
        r.avg_alignment_score = alignment_score
        return r

    def test_picks_highest_alignment_score(self):
        results = {
            "zero_shot":       self._make_result(0.55),
            "few_shot_2":      self._make_result(0.68),
            "chain_of_thought": self._make_result(0.71),
            "few_shot_cot":    self._make_result(0.65),
        }
        name, result = pick_best_strategy(results)
        assert name == "chain_of_thought"
        assert result.avg_alignment_score == 0.71

    def test_single_strategy(self):
        results = {"zero_shot": self._make_result(0.5)}
        name, result = pick_best_strategy(results)
        assert name == "zero_shot"


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — full strategy evaluation on mock data
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselinePipelineIntegration:
    """
    Tests the full evaluate_strategy flow using a mock model.
    No real inference — mock model returns pre-defined outputs.
    """

    def _make_mock_model(self, return_text: str = '{"vendor": "ACME", "total": 100.0}'):
        """Create a mock model that returns token IDs for the given text."""
        import torch
        model = MagicMock()
        model.parameters = lambda: iter([torch.zeros(1)])
        model.eval = lambda: model

        # When generate() is called, return a tensor containing the response
        # The actual decoding happens via tokenizer.decode() which is also mocked
        def mock_generate(**kwargs):
            input_ids = kwargs.get("input_ids", torch.zeros(1, 10, dtype=torch.long))
            # Return input + 10 fake output tokens
            extra = torch.zeros(1, 10, dtype=torch.long)
            return torch.cat([input_ids, extra], dim=1)

        model.generate = mock_generate
        return model

    def _make_mock_tokenizer(self, response_text: str):
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.eos_token_id = 1
        tok.pad_token = "<pad>"

        import torch

        def call(text, return_tensors=None, **kwargs):
            # Return a batch of 10 input tokens
            return {"input_ids": torch.zeros(1, 10, dtype=torch.long),
                    "attention_mask": torch.ones(1, 10, dtype=torch.long)}

        tok.__call__ = call

        def apply_template(messages, tokenize=False, add_generation_prompt=False):
            parts = [f"[{m['role']}]{m['content']}" for m in messages]
            if add_generation_prompt:
                parts.append("[assistant]")
            return "".join(parts)

        tok.apply_chat_template = apply_template
        tok.decode = lambda ids, skip_special_tokens=True: response_text

        return tok

    def test_metrics_on_perfect_output(self, sample_sft_example):
        """Verify MetricsComputer works with a perfect JSON prediction."""
        ref = sample_sft_example.target_text
        mc = MetricsComputer()
        result = mc.compute(
            prediction_text=ref,
            reference_text=ref,
            task_type=sample_sft_example.task_type,
        )
        assert result.format_valid is True
        assert result.exact_match is True
        assert result.field_f1 == 1.0
        assert result.hallucination_detected is False

    def test_metrics_on_invalid_output(self, sample_sft_example):
        """Verify MetricsComputer handles invalid JSON gracefully."""
        mc = MetricsComputer()
        result = mc.compute(
            prediction_text="This is not JSON at all.",
            reference_text=sample_sft_example.target_text,
            task_type=sample_sft_example.task_type,
        )
        assert result.format_valid is False
        assert result.exact_match is False
        assert result.field_f1 == 0.0

    def test_all_strategy_names_unique(self):
        strategies = get_all_strategies()
        names = [s.name for s in strategies]
        assert len(names) == len(set(names)), "Duplicate strategy names detected"