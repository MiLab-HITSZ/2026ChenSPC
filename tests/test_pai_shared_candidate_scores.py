from analyze_candidate_baseline_sweep import _method_block, _replay_pai


def test_pai_reuses_equivalent_image_no_image_scores():
    record = {
        "cp_vbc": {
            "baseline_key": "yes",
            "baseline_pred": "Final answer: yes",
            "candidates": [
                {"key": "yes", "logp_image": -2.0, "logp_prior": -0.5},
                {"key": "no", "logp_image": -2.1, "logp_prior": -3.0},
            ],
        }
    }
    block = _method_block(record, "pai")
    assert block is not None
    assert block["candidates"][0]["logp_no_image"] == -0.5
    assert _replay_pai(
        block,
        {"gamma": 1.5, "beta": 0.1, "gate": "always"},
    ) == "no"


def test_native_pai_block_takes_precedence():
    native = {"baseline_key": "yes", "candidates": []}
    assert _method_block({"pai": native, "cp_vbc": {}}, "pai") is native
