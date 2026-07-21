from analyze_third_model_mc import alignment_audit


def test_alignment_audit_conditions_on_native_cf_errors():
    rows = [
        {
            "_dataset": "test_mc",
            "_task": "mc",
            "side": "counterfactual",
            "gt": "B",
            "cs_gt": "A",
            "cp_vbc": {
                "baseline_key": "A",
                "baseline_pred": "A",
                "prior_top": "A",
            },
        },
        {
            "_dataset": "test_mc",
            "_task": "mc",
            "side": "counterfactual",
            "gt": "C",
            "cs_gt": "A",
            "cp_vbc": {
                "baseline_key": "C",
                "baseline_pred": "C",
                "prior_top": "A",
            },
        },
    ]
    result = alignment_audit(rows)
    assert result["native_cf_errors"] == 1
    assert result["native_error_equals_prior_top_rate"] == 1.0
    assert result["prior_top_equals_paired_cs_answer_rate"] == 1.0
