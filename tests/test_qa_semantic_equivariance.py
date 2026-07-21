from analyze_qa_semantic_equivariance import (
    classify_prior_pattern,
    conditioned_metrics,
    template_rows,
)


def row(pair_id, variant, side, prior, gt, baseline):
    return {
        "pair_id": pair_id,
        "_dataset": f"test_contrastive_{variant}",
        "side": side,
        "gt": gt,
        "cp_vbc": {
            "prior_top": prior,
            "baseline_key": baseline,
            "baseline_pred": baseline,
        },
    }


def complete_pair(pair_id, left_prior, right_prior):
    rows = []
    for variant, prior, gt in (
        ("cs_first", left_prior, "yes"),
        ("cf_first", right_prior, "no"),
    ):
        for side in ("counterfactual", "commonsense"):
            rows.append(row(pair_id, variant, side, prior, gt, "no"))
    return rows


def test_prior_pattern_requires_correct_orientation_not_only_complementarity():
    aligned_rows = complete_pair("p1", "yes", "no")
    inverted_rows = complete_pair("p2", "no", "yes")
    rows = aligned_rows + inverted_rows
    lookup = template_rows(rows, "contrastive")
    aligned = classify_prior_pattern(rows, lookup["p1"])
    inverted = classify_prior_pattern(rows, lookup["p2"])
    assert aligned["aligned_equivariant"] is True
    assert aligned["complementary"] is True
    assert inverted["aligned_equivariant"] is False
    assert inverted["complementary"] is True


def test_conditioned_metrics_pool_both_claim_orders_by_pair():
    rows = complete_pair("p1", "yes", "no")
    lookup = template_rows(rows, "contrastive")
    predictions = ["yes", "yes", "no", "no"]
    metrics = conditioned_metrics(rows, predictions, lookup, {"p1"}, "test")
    assert metrics["n_pairs"] == 1
    assert metrics["pooled"]["counterfactual"]["n_rows"] == 2
    assert metrics["pooled"]["counterfactual"]["repairs"] == 1
