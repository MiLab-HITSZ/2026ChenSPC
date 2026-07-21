import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_qa_template_stress import (
    aggregate_range,
    exact_mcnemar,
    holm_adjust,
    mapping_prediction,
)


def test_exact_mcnemar_is_two_sided_and_handles_ties():
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(5, 0) == 0.0625
    assert exact_mcnemar(3, 2) == 1.0


def test_holm_adjustment_controls_the_ordered_family():
    assert holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5}) == {
        "a": 0.03,
        "b": 0.04,
        "c": 0.5,
    }


def test_aggregate_range_preserves_direction_specific_extrema():
    per_template = {
        "one": {
            "by_variant": {
                "cs_first": {
                    "counterfactual": {"delta": 0.2},
                    "commonsense": {"delta": -0.1},
                },
                "cf_first": {
                    "counterfactual": {"delta": 0.1},
                    "commonsense": {"delta": 0.0},
                },
            }
        },
        "two": {
            "by_variant": {
                "cs_first": {
                    "counterfactual": {"delta": 0.4},
                    "commonsense": {"delta": -0.2},
                },
                "cf_first": {
                    "counterfactual": {"delta": -0.1},
                    "commonsense": {"delta": 0.1},
                },
            }
        },
    }
    result = aggregate_range(per_template)
    assert result["cs_first"]["counterfactual"]["min_delta"] == 0.2
    assert result["cs_first"]["counterfactual"]["max_delta"] == 0.4
    assert result["cs_first"]["counterfactual"]["all_nonnegative"]
    assert not result["cf_first"]["counterfactual"]["all_nonnegative"]


def test_direction_only_mappings_are_explicit():
    assert mapping_prediction("identity", "yes") == "yes"
    assert mapping_prediction("always_yes", "no") == "yes"
    assert mapping_prediction("always_no", "yes") == "no"
    assert mapping_prediction("invert", "yes") == "no"
    assert mapping_prediction("invert", "no") == "yes"
