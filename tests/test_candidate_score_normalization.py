from analyze_candidate_score_normalization import (
    fixed_residual_agreement,
    length_audit,
    mean_score_rows,
)


def scored_row(lengths=(2, 2)):
    candidates = []
    for key, score, length in zip(("a", "b"), (-4.0, -6.0), lengths):
        candidates.append(
            {
                "key": key,
                "logp_image": score,
                "avg_logp_image": score / length,
                "logp_prior": score - 1.0,
                "avg_logp_prior": (score - 1.0) / length,
                "image_token_count": length,
                "prior_token_count": length,
            }
        )
    return {"_task": "mc", "cp_vbc": {"candidates": candidates}}


def test_mean_score_rows_uses_recorded_per_token_scores_without_mutating_source():
    source = [scored_row()]
    transformed = mean_score_rows(source)
    assert transformed[0]["cp_vbc"]["candidates"][0]["logp_image"] == -2.0
    assert source[0]["cp_vbc"]["candidates"][0]["logp_image"] == -4.0


def test_equal_candidate_lengths_make_fixed_residual_ranking_invariant():
    source = [scored_row()]
    transformed = mean_score_rows(source)
    result = fixed_residual_agreement(source, transformed, (0.1, 0.5, 1.0))
    assert result["agreement"] == 1.0
    assert result["disagreements"] == 0


def test_length_audit_detects_unequal_candidate_lengths():
    result = length_audit([scored_row((1, 2))])
    assert result["rows_with_unequal_candidate_lengths"] == 1
