from unittest.mock import patch

from cprc_vlm_runtime import ModelSpec
from evaluate_visual_counterfact_candidate_scores import score


def test_score_accepts_multitoken_candidate_branches():
    spec = ModelSpec(
        name="LLaVA",
        backend="transformers",
        model="llava",
        temperature=0.0,
        max_tokens=1,
        models_root="models",
    )
    item = {"question": "Question", "candidate_keys": ["A", "B"]}
    image_scores = [
        {"logprob": -0.2, "avg_logprob": -0.1, "token_count": 2, "token_ids": [1, 2]},
        {"logprob": -0.4, "avg_logprob": -0.2, "token_count": 2, "token_ids": [1, 3]},
    ]
    prior_scores = [
        {"logprob": -0.1, "avg_logprob": -0.05, "token_count": 2, "token_ids": [1, 2]},
        {"logprob": -0.8, "avg_logprob": -0.4, "token_count": 2, "token_ids": [1, 3]},
    ]
    with patch(
        "evaluate_visual_counterfact_candidate_scores.local_candidate_bundle",
        return_value=(object(), object(), "llava_next"),
    ), patch(
        "evaluate_visual_counterfact_candidate_scores._candidate_logprobs_shared_prefix",
        side_effect=[image_scores, prior_scores],
    ):
        rows, _ = score(spec, item, object())

    assert [row["image_token_count"] for row in rows] == [2, 2]
    assert rows[0]["image_token_ids"] == [1, 2]
    assert rows[1]["logp_prior"] == -0.8
