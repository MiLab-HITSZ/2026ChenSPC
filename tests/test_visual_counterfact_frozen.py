from prepare_visual_counterfact_frozen import answer_text, option_layout, question


def test_answer_text_normalizes_official_color_lists():
    assert answer_text("['green']") == "green"
    assert answer_text("['brown', 'white']") == "brown or white"
    assert answer_text("tree") == "tree"


def test_option_layout_balances_and_remaps_both_answers():
    even = option_layout("color", 0, "green", "purple")
    odd = option_layout("color", 1, "white", "orange")
    assert even["typical_key"] == "A"
    assert even["counterfactual_key"] == "B"
    assert odd["typical_key"] == "B"
    assert odd["counterfactual_key"] == "A"


def test_size_uses_the_opposite_parity_offset():
    layout = option_layout("size", 0, "tree", "Doberman")
    assert layout["typical_key"] == "B"
    assert layout["counterfactual_key"] == "A"
    assert question("size", "Doberman", layout["options"]).endswith(
        "Answer with A or B."
    )
