import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_qa_template_stress import TEMPLATES, option_map, transform


def source_item():
    return {
        "pair_id": "Pair 1",
        "category": "Counting Anomalies",
        "subcategory": "Body Parts",
        "multiple_choice": {
            "question": "How many fingers are visible?",
            "options": ["A. five", "B. six"],
            "counterfactual_gt": "B",
            "commonsense_gt": "A",
        },
        "direct_qa": {},
    }


def test_all_templates_share_one_truth_condition():
    for template_id in TEMPLATES:
        ordinary = transform(source_item(), template_id, "cs_first")
        atypical = transform(source_item(), template_id, "cf_first")
        assert ordinary["direct_qa"]["counterfactual_gt"] == "no"
        assert atypical["direct_qa"]["counterfactual_gt"] == "yes"
        assert ordinary["qa_template_stress"]["truth_condition"] == "first_option_is_correct"
        assert atypical["qa_template_stress"]["truth_condition"] == "first_option_is_correct"


def test_swapped_variants_only_reverse_answer_order():
    for template_id in TEMPLATES:
        ordinary = transform(source_item(), template_id, "cs_first")
        atypical = transform(source_item(), template_id, "cf_first")
        left = ordinary["qa_template_stress"]
        right = atypical["qa_template_stress"]
        assert left["first_option_text"] == right["second_option_text"]
        assert left["second_option_text"] == right["first_option_text"]


def test_templates_do_not_expose_evaluation_side_names():
    for template_id in TEMPLATES:
        row = transform(source_item(), template_id, "cs_first")
        question = row["direct_qa"]["question"].casefold()
        assert "counterfactual" not in question
        assert "commonsense" not in question


def test_combined_multiple_choice_string_is_split_without_separators():
    parsed = option_map(
        ["A. two ears \u2022 B. four ears \u2022 C. six ears \u2022 D. eight ears"]
    )
    assert parsed == {
        "A": "two ears",
        "B": "four ears",
        "C": "six ears",
        "D": "eight ears",
    }
