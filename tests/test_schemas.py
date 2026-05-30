"""
Unit tests for all alignment pipeline schemas.

Run:
    pytest tests/test_schemas.py -v
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.data.schemas import (
    AlignmentSignal,
    BenchmarkResult,
    Conversation,
    DatasetMetadata,
    DatasetSplit,
    DifficultyLevel,
    DPOExample,
    EvaluationResult,
    ExperimentConfig,
    FailureMode,
    GenerationConfig,
    MessageTurn,
    ModelCheckpointMetadata,
    ModelPrediction,
    ModelStage,
    RejectionStrategy,
    Role,
    SFTExample,
    ConversationQuality,
    TaskType,
)


# ─── MessageTurn ──────────────────────────────────────────────────────────────

class TestMessageTurn:
    def test_valid_system_turn(self):
        t = MessageTurn(role=Role.SYSTEM, content="You are a helpful assistant.")
        assert t.role == Role.SYSTEM

    def test_whitespace_only_content_raises(self):
        with pytest.raises(Exception, match="whitespace"):
            MessageTurn(role=Role.USER, content="   ")

    def test_empty_content_raises(self):
        with pytest.raises(Exception):
            MessageTurn(role=Role.ASSISTANT, content="")


# ─── Conversation ─────────────────────────────────────────────────────────────

class TestConversation:
    def _make_conv(self, extra_turns=None):
        turns = [
            MessageTurn(role=Role.SYSTEM, content="You are helpful.", turn_idx=0),
            MessageTurn(role=Role.USER, content="Hello.", turn_idx=1),
            MessageTurn(role=Role.ASSISTANT, content="Hi there!", turn_idx=2),
        ]
        if extra_turns:
            turns.extend(extra_turns)
        return Conversation(turns=turns)

    def test_valid_conversation(self):
        conv = self._make_conv()
        assert conv.system_prompt == "You are helpful."
        assert conv.final_response == "Hi there!"

    def test_first_turn_must_be_system(self):
        with pytest.raises(Exception, match="system"):
            Conversation(turns=[
                MessageTurn(role=Role.USER, content="Hello"),
                MessageTurn(role=Role.ASSISTANT, content="Hi"),
            ])

    def test_prompt_only_excludes_final_assistant(self):
        conv = self._make_conv()
        prompt = conv.prompt_only
        assert all(t.role != Role.ASSISTANT for t in prompt)

    def test_no_user_turn_raises(self):
        with pytest.raises(Exception, match="user"):
            Conversation(turns=[
                MessageTurn(role=Role.SYSTEM, content="System"),
                MessageTurn(role=Role.ASSISTANT, content="Response"),
            ])

    def test_num_user_turns(self):
        conv = self._make_conv()
        assert conv.num_user_turns == 1


# ─── SFTExample ───────────────────────────────────────────────────────────────

class TestSFTExample:
    def _make_example(self, response="The answer is 42."):
        turns = [
            MessageTurn(role=Role.SYSTEM, content="Answer questions.", turn_idx=0),
            MessageTurn(role=Role.USER, content="What is the answer?", turn_idx=1),
            MessageTurn(role=Role.ASSISTANT, content=response, turn_idx=2),
        ]
        return SFTExample(
            conversation=Conversation(turns=turns),
            task_type=TaskType.QUESTION_ANSWERING,
            difficulty=DifficultyLevel.EASY,
            quality=ConversationQuality.SYNTHETIC_TEMPLATE,
            source="test_generator",
        )

    def test_example_id_generated(self):
        ex = self._make_example()
        assert len(ex.example_id) == 36  # UUID4 format

    def test_input_text_returns_user_content(self):
        ex = self._make_example()
        assert ex.input_text == "What is the answer?"

    def test_target_text_returns_assistant_content(self):
        ex = self._make_example()
        assert ex.target_text == "The answer is 42."

    def test_system_prompt_property(self):
        ex = self._make_example()
        assert ex.system_prompt == "Answer questions."

    def test_to_dict_serialisable(self):
        ex = self._make_example()
        d = ex.to_dict()
        json.dumps(d)  # Must not raise

    def test_roundtrip_serialisation(self):
        ex = self._make_example()
        d = ex.to_dict()
        restored = SFTExample.from_dict(d)
        assert restored.input_text == ex.input_text
        assert restored.target_text == ex.target_text
        assert restored.task_type == ex.task_type

    def test_weight_bounds(self):
        with pytest.raises(Exception):
            self._make_example()
            turns = [
                MessageTurn(role=Role.SYSTEM, content="S", turn_idx=0),
                MessageTurn(role=Role.USER, content="U", turn_idx=1),
                MessageTurn(role=Role.ASSISTANT, content="A", turn_idx=2),
            ]
            SFTExample(
                conversation=Conversation(turns=turns),
                task_type=TaskType.OPEN_ENDED,
                weight=15.0,  # Out of bounds
            )


# ─── DPOExample ───────────────────────────────────────────────────────────────

class TestDPOExample:
    def _make_pair(self, chosen="Good answer.", rejected="Bad answer."):
        return DPOExample(
            prompt="<|system|>Help<|user|>Question<|assistant|>",
            chosen=chosen,
            rejected=rejected,
            preference_strength=0.9,
            rejection_strategy=RejectionStrategy.HALLUCINATED_FIELDS,
            task_type=TaskType.INSTRUCTION_FOLLOWING,
        )

    def test_valid_pair(self):
        pair = self._make_pair()
        assert pair.chosen != pair.rejected

    def test_identical_chosen_rejected_raises(self):
        with pytest.raises(Exception, match="identical"):
            self._make_pair(chosen="Same answer.", rejected="Same answer.")

    def test_to_trl_dict_has_required_keys(self):
        pair = self._make_pair()
        trl = pair.to_trl_dict()
        assert set(trl.keys()) == {"prompt", "chosen", "rejected"}

    def test_preference_strength_bounds(self):
        with pytest.raises(Exception):
            self._make_pair()
            DPOExample(
                prompt="P",
                chosen="C",
                rejected="R",
                preference_strength=1.5,  # Out of bounds
            )

    def test_to_dict_serialisable(self):
        pair = self._make_pair()
        d = pair.to_dict()
        json.dumps(d)  # Must not raise

    def test_pair_id_generated(self):
        pair = self._make_pair()
        assert len(pair.pair_id) == 36


# ─── ModelPrediction ──────────────────────────────────────────────────────────

class TestModelPrediction:
    def _make_pred(self, generated="Output text."):
        return ModelPrediction(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            model_stage=ModelStage.SFT,
            prompt="<|system|>...<|user|>Question<|assistant|>",
            generated_text=generated,
            reference_text="Expected output.",
            task_type=TaskType.INSTRUCTION_FOLLOWING,
            input_tokens=100,
            output_tokens=50,
            latency_ms=350.0,
        )

    def test_tokens_per_second(self):
        pred = self._make_pred()
        tps = pred.tokens_per_second
        assert tps > 0

    def test_is_empty_false_for_real_output(self):
        pred = self._make_pred()
        assert not pred.is_empty

    def test_is_empty_true_for_empty_output(self):
        pred = self._make_pred(generated="  ")
        assert pred.is_empty

    def test_to_dict_serialisable(self):
        pred = self._make_pred()
        json.dumps(pred.to_dict())


# ─── EvaluationResult ─────────────────────────────────────────────────────────

class TestEvaluationResult:
    def _make_result(self, format_valid=True, instruction_followed=True,
                     hallucination_detected=False):
        return EvaluationResult(
            model_stage=ModelStage.SFT,
            task_type=TaskType.STRUCTURED_EXTRACTION,
            difficulty=DifficultyLevel.MEDIUM,
            format_valid=format_valid,
            instruction_followed=instruction_followed,
            hallucination_detected=hallucination_detected,
            field_f1=0.88,
            rouge_l=0.72,
            latency_ms=320.0,
        )

    def test_passed_requires_format_valid(self):
        result = self._make_result(format_valid=False)
        assert not result.passed

    def test_passed_fails_on_hallucination(self):
        result = self._make_result(hallucination_detected=True)
        assert not result.passed

    def test_alignment_score_in_range(self):
        result = self._make_result()
        assert 0.0 <= result.alignment_score <= 1.0

    def test_alignment_score_higher_when_better(self):
        good = self._make_result(format_valid=True, instruction_followed=True)
        bad = self._make_result(format_valid=False, instruction_followed=False)
        assert good.alignment_score > bad.alignment_score

    def test_to_dict_serialisable(self):
        result = self._make_result()
        json.dumps(result.to_dict())

    def test_failure_modes_list(self):
        result = EvaluationResult(
            model_stage=ModelStage.BASE,
            task_type=TaskType.TOOL_CALLING,
            difficulty=DifficultyLevel.HARD,
            format_valid=False,
            failure_modes=[FailureMode.FORMAT_ERROR, FailureMode.TRUNCATED_OUTPUT],
        )
        assert len(result.failure_modes) == 2


# ─── AlignmentSignal ──────────────────────────────────────────────────────────

class TestAlignmentSignal:
    def test_composite_score_range(self):
        signal = AlignmentSignal(
            safety_score=0.9,
            helpfulness_score=0.8,
            honesty_score=0.7,
            instruction_score=0.85,
        )
        assert 0.0 <= signal.composite_score <= 1.0

    def test_harmful_flag(self):
        signal = AlignmentSignal(is_harmful=True, safety_score=0.1)
        assert signal.is_harmful


# ─── BenchmarkResult ──────────────────────────────────────────────────────────

class TestBenchmarkResult:
    def _make_result(self, stage, format_valid=0.7, hallucination=0.2):
        return BenchmarkResult(
            model_id="test-model",
            model_stage=stage,
            n_examples=100,
            format_valid=format_valid,
            instruction_followed=0.75,
            hallucination_rate=hallucination,
            avg_alignment_score=0.65,
            avg_latency_ms=350.0,
            p95_latency_ms=600.0,
        )

    def test_summary_string(self):
        result = self._make_result(ModelStage.SFT)
        summary = result.summary()
        assert "SFT" in summary
        assert "%" in summary

    def test_delta_vs_positive_improvement(self):
        baseline = self._make_result(ModelStage.BASE, format_valid=0.6)
        improved = self._make_result(ModelStage.SFT, format_valid=0.85)
        deltas = improved.delta_vs(baseline)
        assert deltas["format_valid_delta_pp"] > 0

    def test_delta_vs_hallucination_improvement(self):
        baseline = self._make_result(ModelStage.BASE, hallucination=0.3)
        improved = self._make_result(ModelStage.SFT, hallucination=0.08)
        deltas = improved.delta_vs(baseline)
        assert deltas["hallucination_rate_delta_pp"] > 0  # Positive = improvement

    def test_to_dict_serialisable(self):
        result = self._make_result(ModelStage.DPO)
        json.dumps(result.to_dict())


# ─── ExperimentConfig ─────────────────────────────────────────────────────────

class TestExperimentConfig:
    def _make_config(self):
        return ExperimentConfig(
            run_name="sft-qwen2.5-7b-r16",
            model_stage=ModelStage.SFT,
            base_model_id="Qwen/Qwen2.5-7B-Instruct",
            lora_rank=16,
            lora_alpha=32,
            quantization_bits=4,
            learning_rate=2e-4,
            per_device_batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=3.0,
            train_size=7000,
            gpu_name="NVIDIA A100 40GB",
        )

    def test_effective_batch_size(self):
        cfg = self._make_config()
        assert cfg.effective_batch_size == 4 * 4 * 1  # batch * accum * gpus

    def test_to_wandb_dict_flat(self):
        cfg = self._make_config()
        d = cfg.to_wandb_dict()
        # W&B dict must be flat — no nested dicts
        for v in d.values():
            assert not isinstance(v, dict), f"Nested dict found in W&B config"

    def test_to_wandb_dict_serialisable(self):
        cfg = self._make_config()
        json.dumps(cfg.to_wandb_dict())

    def test_lora_scale_computed(self):
        cfg = self._make_config()
        d = cfg.to_wandb_dict()
        assert d["lora_scale"] == 32 / 16  # alpha / rank = 2.0


# ─── ModelCheckpointMetadata ──────────────────────────────────────────────────

class TestModelCheckpointMetadata:
    def test_valid_checkpoint(self):
        ckpt = ModelCheckpointMetadata(
            model_stage=ModelStage.SFT,
            base_model_id="Qwen/Qwen2.5-7B-Instruct",
            adapter_path="experiments/sft_runs/run_001/checkpoint-500",
            training_step=500,
            training_epoch=1.5,
            eval_loss=0.312,
            trainable_params_M=40.2,
            lora_rank=16,
            is_best=True,
        )
        assert ckpt.is_best
        d = ckpt.to_dict()
        json.dumps(d)

    def test_checkpoint_id_generated(self):
        ckpt = ModelCheckpointMetadata(
            model_stage=ModelStage.DPO,
            base_model_id="test",
            adapter_path="test/path",
            training_step=100,
            training_epoch=0.5,
        )
        assert len(ckpt.checkpoint_id) == 36