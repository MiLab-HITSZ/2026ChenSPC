from analyze_spc_mc_option_permutation import (
    holm_adjust,
    paired_method_test,
    paired_variant_test,
    semantic_consistency,
    semantic_key,
)


def test_semantic_key_maps_new_letter_back_to_original_option():
    assert semantic_key("A", [1, 2, 3, 0]) == "B"
    assert semantic_key("D", [1, 2, 3, 0]) == "A"


def test_semantic_consistency_ignores_letter_changes_that_follow_options():
    rows = [
        {"pair_id": "Pair 1", "side": "counterfactual", "gt": "A"},
        {"pair_id": "Pair 1", "side": "counterfactual", "gt": "D"},
    ]
    predictions = ["A", "D"]
    result = semantic_consistency(
        rows,
        predictions,
        {"original": [0], "rotate_1": [1]},
        {"original": [0, 1, 2, 3], "rotate_1": [1, 2, 3, 0]},
    )
    assert result["counterfactual"]["semantic_consistency"] == 1.0


def test_paired_variant_comparison_matches_rows_by_pair_and_side():
    rows = [
        {"_task": "mc", "pair_id": "Pair 1", "side": "counterfactual", "gt": "A"},
        {"_task": "mc", "pair_id": "Pair 1", "side": "counterfactual", "gt": "D"},
    ]
    result = paired_variant_test(
        rows, ["B", "D"], [0], [1], "counterfactual"
    )
    assert result["candidate_repairs_over_reference"] == 1
    assert result["candidate_harms_vs_reference"] == 0


def test_paired_method_comparison_uses_same_variant_rows():
    rows = [
        {"_task": "mc", "pair_id": "Pair 1", "side": "counterfactual", "gt": "A"},
        {"_task": "mc", "pair_id": "Pair 2", "side": "counterfactual", "gt": "B"},
    ]
    result = paired_method_test(
        rows, ["C", "B"], ["A", "D"], [0, 1], "counterfactual"
    )
    assert result["candidate_repairs_over_reference"] == 1
    assert result["candidate_harms_vs_reference"] == 1


def test_holm_adjustment_is_monotone_in_ordered_p_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.5}
