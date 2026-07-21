from analyze_cprc_paired_calibration import VariantSelector, endpoint_metrics


def row(task, side, gt, baseline):
    return {
        "_dataset": f"dev_{task}",
        "_task": task,
        "side": side,
        "gt": gt,
        "cp_vbc": {"baseline_key": baseline, "baseline_pred": baseline},
    }


def test_paired_selector_rejects_cf_gain_that_breaks_cs_budget():
    rows = []
    for task in ("mc", "qa"):
        rows.extend(
            [
                row(task, "counterfactual", "b", "a"),
                row(task, "commonsense", "a", "a"),
            ]
        )
    paired = VariantSelector(rows, "paired_calibration", 0.04, 0.5, True)
    cf_only = VariantSelector(rows, "paired_fit_cf_objective", 0.04, 0.5, True)
    aggressive = ["b", "b", "b", "b"]
    identity = ["a", "a", "a", "a"]
    paired.consider({"name": "aggressive"}, aggressive)
    paired.consider({"name": "identity"}, identity)
    cf_only.consider({"name": "aggressive"}, aggressive)
    cf_only.consider({"name": "identity"}, identity)
    assert paired.best.params["name"] == "identity"
    assert cf_only.best.params["name"] == "aggressive"


def test_endpoint_metrics_reports_repairs_and_harms_by_task_and_side():
    rows = [
        row("mc", "counterfactual", "b", "a"),
        row("mc", "commonsense", "a", "a"),
    ]
    metrics = endpoint_metrics(rows, ["b", "b"], range(2))
    assert metrics["mc"]["counterfactual"]["repairs"] == 1
    assert metrics["mc"]["counterfactual"]["harms"] == 0
    assert metrics["mc"]["commonsense"]["repairs"] == 0
    assert metrics["mc"]["commonsense"]["harms"] == 1


def test_selector_supports_mc_only_third_model_protocol():
    rows = [
        row("mc", "counterfactual", "b", "a"),
        row("mc", "commonsense", "a", "a"),
    ]
    selector = VariantSelector(
        rows,
        "paired_calibration",
        0.04,
        0.5,
        True,
        tasks=("mc",),
    )
    selector.consider({"name": "identity"}, ["a", "a"])
    assert selector.best is not None
    assert set(selector.best.development_metrics) == {"mc"}


def test_selector_can_report_a_heldout_subset():
    rows = [
        row("mc", "counterfactual", "b", "a"),
        row("mc", "commonsense", "a", "a"),
        {**row("mc", "counterfactual", "b", "a"), "_dataset": "test_mc"},
        {**row("mc", "commonsense", "a", "a"), "_dataset": "test_mc"},
    ]
    selector = VariantSelector(
        rows,
        "paired_calibration",
        0.04,
        0.5,
        True,
        tasks=("mc",),
        selection_ids=(0, 1),
    )
    selector.consider({"name": "identity"}, ["a", "a", "a", "a"])
    result = selector.serialize(test_ids=(2,))
    assert result["test_metrics"]["mc"]["counterfactual"]["n"] == 1
    assert result["test_metrics"]["mc"]["commonsense"]["n"] == 0
