import copy

import numpy as np

from analyze_hierarchical_eb_lambda_cv import context_features
from analyze_spc_robustness import (
    choose_development_pairs,
    derangement,
    mean_prior_rows,
    permuted_prior_rows,
    prior_by_key,
    relabel_split,
    shuffled_prior_rows,
)
from analyze_pope_native_spc import binary_metrics, exact_mcnemar, multinomial_bootstrap_ci


def make_row(pair_id, dataset, task="qa", subcategory="Color", prior=(2.0, -1.0)):
    return {
        "pair_id": pair_id,
        "subcategory": subcategory,
        "side": "counterfactual",
        "task": task,
        "_task": task,
        "_dataset": dataset,
        "gt": "no",
        "cp_vbc": {
            "baseline_key": "yes",
            "candidates": [
                {"key": "yes", "logp_image": 0.7, "logp_prior": prior[0]},
                {"key": "no", "logp_image": 0.1, "logp_prior": prior[1]},
            ],
        },
    }


def test_image_only_context_is_independent_of_prior_scores():
    first = make_row("Pair 1", "dev_qa", prior=(9.0, -4.0))
    second = make_row("Pair 1", "dev_qa", prior=(-100.0, 30.0))
    assert np.allclose(
        context_features(first, "image_only"),
        context_features(second, "image_only"),
    )
    assert not np.allclose(context_features(first), context_features(second))


def test_mean_prior_uses_development_pool_only():
    rows = [
        make_row("Pair 1", "dev_qa", prior=(4.0, 0.0)),
        make_row("Pair 2", "dev_qa", prior=(2.0, 0.0)),
        make_row("Pair 3", "test_qa", prior=(-500.0, 500.0)),
    ]
    transformed = mean_prior_rows(rows)
    assert prior_by_key(transformed[2]) == {"yes": 1.5, "no": -1.5}


def test_shuffled_prior_donors_are_development_rows():
    rows = [
        make_row("Pair 1", "dev_qa", prior=(4.0, 0.0)),
        make_row("Pair 2", "dev_qa", prior=(2.0, 0.0)),
        make_row("Pair 3", "test_qa", prior=(-500.0, 500.0)),
    ]
    _, donors = shuffled_prior_rows(rows, 7)
    assert all("Pair 3" not in donor for donor in donors.values())
    assert donors["qa/Pair 1/counterfactual"] != "qa/Pair 1/counterfactual"


def test_permutation_is_non_identity_and_preserves_values():
    rows = [make_row("Pair 1", "dev_qa", prior=(4.0, 0.0))]
    transformed = permuted_prior_rows(rows, 11)
    assert prior_by_key(transformed[0]) == {"yes": -2.0, "no": 2.0}
    assert sorted(derangement(4, __import__("random").Random(3))) == [0, 1, 2, 3]


def test_stratified_split_keeps_pair_tasks_and_sides_grouped():
    rows = []
    for subcategory in ("Color", "Relation"):
        for pair in range(1, 7):
            for task in ("qa", "mc"):
                for side in ("counterfactual", "commonsense"):
                    row = make_row(
                        f"{subcategory}-{pair}",
                        "test_qa",
                        task=task,
                        subcategory=subcategory,
                    )
                    row["side"] = side
                    rows.append(row)
    selected = choose_development_pairs(rows, 2, 17)
    relabeled = relabel_split(copy.deepcopy(rows), selected)
    for subcategory, pair_ids in selected.items():
        assert len(pair_ids) == 2
        labels = {
            row["_dataset"]
            for row in relabeled
            if row["subcategory"] == subcategory and row["pair_id"] in pair_ids
        }
        assert labels == {"dev_qa", "dev_mc"}


def test_pope_native_metrics_match_official_binary_definitions():
    metrics = binary_metrics(
        ["yes", "yes", "no", "no"],
        ["yes", "no", "yes", "no"],
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_pope_joint_statistics_are_paired_and_deterministic():
    assert exact_mcnemar(4, 0) == 0.125
    assert multinomial_bootstrap_ci(3, 1, 10, 0, 17) == [0.2, 0.2]
    first = multinomial_bootstrap_ci(10, 2, 100, 1000, 17)
    second = multinomial_bootstrap_ci(10, 2, 100, 1000, 17)
    assert first == second
    assert first[0] <= 0.08 <= first[1]
