from analyze_conflictvis_spc import _metrics
from prepare_conflictvis_transfer import _parse_options


def test_conflictvis_mc_options_are_parsed_in_order():
    question = (
        "Question: What is shown?\n"
        "Options: (A) first. (B) second. (C) third. (D) fourth.\n"
        "Answer with the option's letter from the given choices directly."
    )
    assert _parse_options(question) == [
        "A. first.",
        "B. second.",
        "C. third.",
        "D. fourth.",
    ]


def test_transfer_metrics_count_repairs_and_harms():
    result = _metrics(
        labels=["yes", "yes", "no", "no"],
        baseline=["no", "yes", "no", "yes"],
        method=["yes", "no", "no", "yes"],
        draws=0,
        seed=7,
    )
    assert result["repairs"] == 1
    assert result["harms"] == 1
    assert result["delta_accuracy"] == 0.0
