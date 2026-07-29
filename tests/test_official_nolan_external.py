from evaluate_official_nolan_external import binary_f1, group_metrics
from reconcile_official_nolan_external import merged_rows, select_fallback_rows


def test_group_metrics_counts_changes_repairs_and_harms():
    rows = [
        {
            "native_key": "yes",
            "pred_key": "no",
            "native_correct": False,
            "correct": True,
        },
        {
            "native_key": "yes",
            "pred_key": "no",
            "native_correct": True,
            "correct": False,
        },
        {
            "native_key": "yes",
            "pred_key": "yes",
            "native_correct": True,
            "correct": True,
        },
    ]

    metrics = group_metrics(rows)

    assert metrics["changed"] == 2
    assert metrics["repairs"] == 1
    assert metrics["harms"] == 1
    assert metrics["delta_accuracy_points"] == 0.0


def test_binary_f1_uses_yes_as_the_positive_class():
    rows = [
        {"gt": "yes", "native_key": "yes", "pred_key": "no"},
        {"gt": "yes", "native_key": "no", "pred_key": "yes"},
        {"gt": "no", "native_key": "no", "pred_key": "yes"},
        {"gt": "no", "native_key": "no", "pred_key": "no"},
    ]

    assert binary_f1(rows, "native_key") == 2 / 3
    assert binary_f1(rows, "pred_key") == 0.5


def test_fallback_selection_and_merge_replace_invalid_decoding():
    identity = {
        "pair_id": "POPE::row-1",
        "task": "qa",
        "side": "counterfactual",
    }
    source = {**identity, "status": "ok", "gt": "no"}
    one_token = {
        **identity,
        "status": "ok",
        "gt": "no",
        "native_pred": "yes",
        "native_key": "yes",
        "native_correct": False,
        "pred": "The",
        "pred_key": None,
        "correct": False,
        "nolan": {"answer_protocol": "single_token"},
    }
    native = {
        **identity,
        "status": "ok",
        "pred": "yes",
        "pred_key": "yes",
        "correct": False,
    }
    nolan = {
        **identity,
        "status": "ok",
        "pred": "Answer: no",
        "pred_key": "no",
        "correct": True,
        "nolan": {"answer_protocol": "full_generation"},
    }

    assert select_fallback_rows([source], [one_token]) == [source]
    merged = merged_rows([one_token], [native], [nolan])[0]
    assert merged["native_key"] == "yes"
    assert merged["pred_key"] == "no"
    assert merged["correct"] is True
    assert merged["decoding_fallback"] == "full_generation"
