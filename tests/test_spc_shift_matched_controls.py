from analyze_spc_shift_matched_controls import (
    aggregate_metric_blocks,
    compare_outcomes,
    relabel_rows,
)


def test_relabel_rows_keeps_pair_and_task_boundaries():
    rows = [
        {
            "_dataset": split,
            "_task": task,
            "pair_id": pair_id,
            "category": category,
        }
        for split, pair_id, category in (
            ("dev_mc", "Pair 1", "Counting"),
            ("dev_mc", "Pair 2", "Attribute"),
            ("test_mc", "Pair 3", "Counting"),
            ("test_mc", "Pair 4", "Attribute"),
        )
        for task in ("mc", "qa")
    ]
    shifted = relabel_rows(
        rows,
        {"Pair 2"},
        lambda row: row["category"] == "Counting",
    )
    labels = {(row["pair_id"], row["_task"], row["_dataset"]) for row in shifted}
    assert ("Pair 2", "mc", "dev_mc") in labels
    assert ("Pair 2", "qa", "dev_qa") in labels
    assert ("Pair 3", "mc", "test_mc") in labels
    assert ("Pair 3", "qa", "test_qa") in labels
    assert ("Pair 1", "mc", "ignore_mc") in labels
    assert ("Pair 4", "qa", "ignore_qa") in labels


def test_aggregate_metric_blocks_weights_by_examples():
    blocks = [
        {
            "test_mc": {
                "counterfactual": {
                    "n": 2,
                    "baseline_correct": 1,
                    "correct": 2,
                    "overrides": 1,
                },
                "commonsense": {
                    "n": 2,
                    "baseline_correct": 2,
                    "correct": 2,
                    "overrides": 0,
                },
            }
        },
        {
            "test_mc": {
                "counterfactual": {
                    "n": 1,
                    "baseline_correct": 0,
                    "correct": 0,
                    "overrides": 0,
                },
                "commonsense": {
                    "n": 1,
                    "baseline_correct": 1,
                    "correct": 0,
                    "overrides": 1,
                },
            }
        },
    ]
    result = aggregate_metric_blocks(blocks)
    assert result["mc"]["counterfactual"]["n"] == 3
    assert result["mc"]["counterfactual"]["delta"] == 1 / 3
    assert result["mc"]["commonsense"]["delta"] == -1 / 3


def test_compare_outcomes_is_oriented_candidate_minus_reference():
    reference = {("mc", "1", "cf"): False, ("mc", "2", "cf"): True}
    candidate = {("mc", "1", "cf"): True, ("mc", "2", "cf"): False}
    result = compare_outcomes(reference, candidate)
    assert result["candidate_repairs_over_reference"] == 1
    assert result["candidate_harms_vs_reference"] == 1
    assert result["candidate_minus_reference_accuracy"] == 0.0
