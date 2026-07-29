import json
import math
from types import SimpleNamespace

import torch

from official_nolan_qwen3vl import (
    OFFICIAL_NOLAN_BETA,
    _forward_branch,
    choose_candidate_with_nolan,
    nolan_adjust_logits,
    nolan_alpha,
    symmetric_kl_from_logits,
)
from evaluate_official_nolan_qwen3vl import (
    answer_stop_token_ids,
    generated_answer_key,
    refresh_native_metadata,
    source_native_answer,
)
from evaluate_official_nolan_candidates import evaluate_trace_replay_row


def test_symmetric_kl_is_zero_for_identical_distributions():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    value = symmetric_kl_from_logits(logits, logits)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-7)


def test_symmetric_kl_is_invariant_to_argument_order():
    first = torch.tensor([[2.0, -1.0, 0.5]])
    second = torch.tensor([[-0.5, 1.0, 0.25]])
    forward = symmetric_kl_from_logits(first, second)
    reverse = symmetric_kl_from_logits(second, first)
    prob_first = torch.softmax(first.float(), dim=-1)
    prob_second = torch.softmax(second.float(), dim=-1)
    official_style = 0.5 * (
        torch.nn.functional.kl_div(
            prob_first.log(), prob_second, reduction="batchmean"
        )
        + torch.nn.functional.kl_div(
            prob_second.log(), prob_first, reduction="batchmean"
        )
    )
    assert torch.allclose(forward, reverse, atol=1e-7)
    assert torch.allclose(forward[0], official_style, atol=1e-7)
    assert float(forward[0]) > 0


def test_dynamic_alpha_matches_published_formula():
    gamma = torch.tensor([0.25, 2.0])
    actual = nolan_alpha(gamma, beta=OFFICIAL_NOLAN_BETA)
    expected = torch.tensor(
        [
            OFFICIAL_NOLAN_BETA * (math.tanh(1.0 / 0.25) + 1.0),
            OFFICIAL_NOLAN_BETA * (math.tanh(1.0 / 2.0) + 1.0),
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-7)


def test_adjusted_logits_use_official_full_vocabulary_rule():
    multimodal = torch.tensor([[3.0, 1.0, -1.0]])
    text_only = torch.tensor([[4.0, -2.0, 0.5]])
    adjusted, gamma, alpha = nolan_adjust_logits(multimodal, text_only, beta=0.8)
    expected = (1.0 + alpha[:, None]) * multimodal - alpha[:, None] * text_only
    assert torch.allclose(adjusted, expected, atol=1e-7)
    assert gamma.shape == (1,)
    assert alpha.shape == (1,)


def test_beta_zero_is_the_matched_native_decoder():
    multimodal = torch.tensor([[3.0, 1.0, -1.0]])
    text_only = torch.tensor([[-2.0, 4.0, 0.5]])
    adjusted, _, alpha = nolan_adjust_logits(multimodal, text_only, beta=0.0)
    assert torch.equal(alpha, torch.zeros_like(alpha))
    assert torch.allclose(adjusted, multimodal.float(), atol=0.0)


def test_identical_streams_preserve_the_token_ranking():
    logits = torch.tensor([[0.25, 2.0, -0.5]])
    adjusted, _, _ = nolan_adjust_logits(logits, logits, beta=0.8)
    assert int(torch.argmax(adjusted, dim=-1)[0]) == int(
        torch.argmax(logits, dim=-1)[0]
    )


def test_forward_branch_requests_only_the_next_token_logits():
    class FakeModel:
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, "logits_to_keep": None}

        def __call__(self, input_ids, logits_to_keep, return_dict):
            assert logits_to_keep == 1
            assert return_dict is True
            return SimpleNamespace(logits=torch.tensor([[[1.0, 2.0, 3.0]]]))

        def _update_model_kwargs_for_generation(
            self, outputs, model_kwargs, is_encoder_decoder
        ):
            assert is_encoder_decoder is False
            return {**model_kwargs, "past_key_values": "cache"}

    logits, state = _forward_branch(
        FakeModel(),
        torch.tensor([[1, 2]]),
        {"use_cache": True},
        first_iteration=True,
    )

    assert logits.shape == (1, 3)
    assert state["past_key_values"] == "cache"


def test_constrained_candidate_choice_uses_full_vocab_adjusted_logits():
    class FakeModel:
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, "logits_to_keep": None}

        def __call__(self, input_ids, logits_to_keep, return_dict):
            assert logits_to_keep == 1
            if int(input_ids[0, 0]) == 10:
                logits = torch.tensor([[[0.0, 3.0, 1.0, -2.0]]])
            else:
                logits = torch.tensor([[[0.0, 4.0, 0.0, -2.0]]])
            return SimpleNamespace(logits=logits)

        def _update_model_kwargs_for_generation(
            self, outputs, model_kwargs, is_encoder_decoder
        ):
            return model_kwargs

    choice = choose_candidate_with_nolan(
        FakeModel(),
        {"input_ids": torch.tensor([[10]])},
        {"input_ids": torch.tensor([[20]])},
        {"A": [1], "B": [2]},
        beta=0.8,
    )

    assert choice.native_key == "A"
    assert choice.prior_key == "A"
    assert choice.key == "B"
    assert choice.diagnostics["distribution"] == "full_vocabulary"


def test_generated_mc_parser_does_not_treat_an_article_as_option_a():
    assert generated_answer_key("mc", "Based on a close examination") is None
    assert generated_answer_key("mc", "Final answer: B") == "B"


def test_answer_stop_tokens_use_the_candidate_final_tokens():
    row = {
        "cp_vbc": {
            "candidates": [
                {"image_token_ids": [141, 59603]},
                {"image_token_ids": [141, 59616]},
            ]
        }
    }
    assert answer_stop_token_ids(row) == [59603, 59616]


def test_generated_qa_parser_requires_an_explicit_answer():
    assert generated_answer_key("qa", "There is no obvious reason to change it.") is None
    assert generated_answer_key("qa", "Answer: no") == "no"


def test_source_native_answer_uses_the_saved_pre_mitigation_prediction():
    row = {
        "task": "mc",
        "gt": "B",
        "pred": "Final answer: B",
        "correct": True,
        "cp_vbc": {
            "baseline_pred": "Final answer: A",
            "baseline_key": "A",
            "final_key": "B",
        },
    }
    assert source_native_answer(row) == ("Final answer: A", "A", False)


def test_refresh_native_metadata_prefers_matched_decoding_control(tmp_path):
    identity = {"pair_id": "Pair 1", "task": "qa", "side": "counterfactual"}
    source = {
        **identity,
        "gt": "no",
        "cp_vbc": {"baseline_pred": "Yes", "baseline_key": "yes"},
    }
    result = {
        **identity,
        "status": "ok",
        "pred": "no",
        "pred_key": "no",
        "correct": True,
    }
    native = {
        **identity,
        "status": "ok",
        "pred": "yes",
        "pred_key": "yes",
        "correct": False,
    }
    output = tmp_path / "result.jsonl"
    output.write_text(json.dumps(result) + "\n", encoding="utf-8")

    refreshed = refresh_native_metadata(output, [source], [native])

    assert refreshed[0]["native_key"] == "yes"
    assert refreshed[0]["native_correct"] is False
    assert refreshed[0]["candidate_native_key"] == "yes"


def test_trace_replay_uses_full_vocabulary_alpha_for_candidate_ranking():
    row = {
        "pair_id": "Pair 1",
        "task": "qa",
        "side": "counterfactual",
        "question": "Is the claim true?",
        "gt": "no",
        "cp_vbc": {
            "scoring_prompt": "Is the claim true?\nAnswer with yes or no.",
            "baseline_pred": "yes",
            "baseline_key": "yes",
            "candidates": [
                {
                    "key": "yes",
                    "text": " yes",
                    "logp_image": -0.1,
                    "logp_prior": -0.01,
                    "image_token_ids": [10],
                    "prior_token_ids": [10],
                },
                {
                    "key": "no",
                    "text": " no",
                    "logp_image": -0.2,
                    "logp_prior": -2.0,
                    "image_token_ids": [20],
                    "prior_token_ids": [20],
                },
            ],
        },
    }
    trace = {
        "nolan": {
            "beta": 0.8,
            "steps": [{"step": 0, "alpha": 1.0, "symmetric_kl": 0.5}],
        }
    }
    spec = SimpleNamespace(name="test", model="test/model")

    result = evaluate_trace_replay_row(spec, row, trace, beta=0.8)

    assert result["native_key"] == "yes"
    assert result["pred_key"] == "no"
    assert result["correct"] is True
    assert result["nolan"]["distribution"] == "full_vocabulary"
