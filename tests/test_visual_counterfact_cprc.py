from analyze_visual_counterfact_cprc import holm_adjust, mechanism_diagnostic


def test_holm_adjust_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.20}


def test_mechanism_diagnostic_conditions_alignment_on_errors():
    rows = [
        {
            "side": "counterfactual",
            "gt": "B",
            "typical_key": "A",
            "cp_vbc": {"baseline_key": "A", "prior_top": "A"},
        },
        {
            "side": "commonsense",
            "gt": "A",
            "typical_key": "A",
            "cp_vbc": {"baseline_key": "A", "prior_top": "A"},
        },
    ]
    result = mechanism_diagnostic(rows, [0, 1])
    assert result["counterfactual"]["baseline_errors"] == 1
    assert result["counterfactual"]["error_matches_prior_top_rate"] == 1.0
    assert result["commonsense"]["prior_top_is_typical_rate"] == 1.0
