from build_mc_option_permutation import canonical_options, option_text, permute_item


def sample_item():
    return {
        "pair_id": "Pair 1",
        "category": "Counting",
        "subcategory": "Body Parts",
        "direct_qa": {"question": "Q?"},
        "captioning": {"question": "Describe."},
        "multiple_choice": {
            "question": "How many?",
            "options": ["A. one", "B. two", "C. three", "D. four"],
            "counterfactual_gt": "B",
            "commonsense_gt": "A",
        },
    }


def test_rotation_preserves_option_semantics_and_remaps_gt():
    result = permute_item(sample_item(), (1, 2, 3, 0), "rotate_1")
    assert result["multiple_choice"]["options"] == [
        "A. two",
        "B. three",
        "C. four",
        "D. one",
    ]
    assert result["multiple_choice"]["counterfactual_gt"] == "A"
    assert result["multiple_choice"]["commonsense_gt"] == "D"
    assert result["direct_qa"] == sample_item()["direct_qa"]


def test_malformed_single_string_options_are_split_before_permutation():
    item = sample_item()
    item["multiple_choice"]["options"] = ["A. one B. two C. three D. four"]
    result = permute_item(item, (3, 2, 1, 0), "reverse")
    assert [option_text(value) for value in result["multiple_choice"]["options"]] == [
        "four",
        "three",
        "two",
        "one",
    ]
    assert result["multiple_choice"]["counterfactual_gt"] == "C"


def test_unlabelled_comma_fragments_are_merged_with_previous_option():
    assert canonical_options(
        ["A. normal", "B. bacteria-sized", "walked by a bacterium", "C. absent", "D. large"]
    ) == [
        "A. normal",
        "B. bacteria-sized, walked by a bacterium",
        "C. absent",
        "D. large",
    ]
