#!/usr/bin/env python3
"""Replay matched SPC/NoLan proposal and gate controls from frozen scores."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from analyze_spc_gate_pareto import apply_gate_family, paired_counts, prepare_geometry
from analyze_spc_robustness import load_model_rows
from analyze_hierarchical_eb_lambda_cv import (
    dataset_metrics,
    fit_lambda_posterior,
    inverse_softplus,
    policy_details_gauss_hermite,
    resolved_baseline_key,
)
from analyze_nolan_anchored_spc import nolan_lambda


FAMILY = "map_no_laplace_or_density"
QUADRATURE_POINTS = 64


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def compare_predictions(
    rows: Sequence[Mapping[str, Any]],
    left: Sequence[str],
    right: Sequence[str],
    indices: Sequence[int],
) -> Dict[str, Any]:
    left_only = right_only = 0
    for index in indices:
        answer = str(rows[index].get("gt", "")).strip().lower()
        left_correct = str(left[index]).lower() == answer
        right_correct = str(right[index]).lower() == answer
        left_only += int(left_correct and not right_correct)
        right_only += int(right_correct and not left_correct)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "exact_mcnemar_p_two_sided": exact_mcnemar(left_only, right_only),
    }


def summarize(
    rows: Sequence[Dict[str, Any]],
    geometry: Any,
    predictions: Sequence[str],
) -> Dict[str, Any]:
    return {
        "test_metrics": dataset_metrics(rows, predictions, geometry.test_ids),
        "test_repairs_harms": paired_counts(rows, predictions, geometry.test_ids),
    }


def run_model(
    model: str,
    spec: Mapping[str, Any],
    nolan_beta: float,
    gate_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = load_model_rows(spec)
    geometry = prepare_geometry(rows)
    selected = gate_artifact["experiments"][model][
        "selected_by_development_cs_budget"
    ][FAMILY]["0.04"]
    params = selected["params"]

    theta, covariance, _ = fit_lambda_posterior(
        rows, geometry.contexts, geometry.train_ids, float(params["l2"])
    )
    spc_details = policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        theta,
        np.zeros_like(covariance),
        lambda_cap=float(params["lambda_cap"]),
        quadrature_points=QUADRATURE_POINTS,
        support_distances=np.zeros(len(rows)),
        arity_supported=geometry.arity_supported,
    )
    spc = apply_gate_family(rows, spc_details, params["gate"], FAMILY)
    spc_proposal_only = [str(detail["proposal"]) for detail in spc_details]

    anchors = np.asarray(
        [nolan_lambda(row, nolan_beta) for row in rows], dtype=np.float64
    )
    latent_offsets = np.asarray(
        [inverse_softplus(value) for value in anchors], dtype=np.float64
    )
    dimension = geometry.contexts.shape[1]
    nolan_details = policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        np.zeros(dimension),
        np.zeros((dimension, dimension)),
        lambda_cap=0.8,
        quadrature_points=QUADRATURE_POINTS,
        support_distances=np.zeros(len(rows)),
        arity_supported=geometry.arity_supported,
        latent_offsets=latent_offsets,
    )
    nolan_raw = [str(detail["proposal"]) for detail in nolan_details]
    nolan_same_gate = apply_gate_family(
        rows, nolan_details, params["gate"], FAMILY
    )
    native = [resolved_baseline_key(row) for row in rows]

    methods = {
        "native": native,
        "spc": spc,
        "spc_proposal_only": spc_proposal_only,
        "nolan_candidate_adaptation": nolan_raw,
        "nolan_candidate_adaptation_same_spc_gate": nolan_same_gate,
    }
    output = {
        "spc_params": params,
        "methods": {
            name: summarize(rows, geometry, predictions)
            for name, predictions in methods.items()
            if name != "native"
        },
        "paired_comparisons": {},
    }
    for task in ("mc", "qa"):
        for side in ("counterfactual", "commonsense"):
            indices = [
                index
                for index in geometry.test_ids
                if rows[index]["_dataset"] == f"test_{task}"
                and rows[index].get("side") == side
            ]
            key = f"test_{task}/{side}"
            output["paired_comparisons"][key] = {
                "spc_vs_native": compare_predictions(
                    rows, spc, native, indices
                ),
                "spc_proposal_only_vs_native": compare_predictions(
                    rows, spc_proposal_only, native, indices
                ),
                "nolan_same_gate_vs_native": compare_predictions(
                    rows, nolan_same_gate, native, indices
                ),
                "spc_vs_nolan_same_gate": compare_predictions(
                    rows, spc, nolan_same_gate, indices
                ),
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/spc_nolan_anchor_v1.json"
    )
    parser.add_argument(
        "--gate-artifact",
        default="result/cprc_robustness/gate_pareto_qwen_llava_v1.json",
    )
    parser.add_argument(
        "--output",
        default="result/paper_revision_stats/matched_spc_nolan_controls.json",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    gate_artifact = json.loads(
        Path(args.gate_artifact).read_text(encoding="utf-8")
    )
    payload = {
        "version": "matched_spc_nolan_controls_v1",
        "selection_split": "development_only",
        "test_refit": False,
        "comparison": (
            "The SPC proposal and the NoLan candidate-score adaptation are "
            "replayed with the exact primary SPC gate; SPC proposal-only "
            "removes that gate without refitting the learned coefficient."
        ),
        "models": {},
    }
    for model, spec in config["models"].items():
        payload["models"][model] = run_model(
            model,
            spec,
            float(config["nolan_beta"]),
            gate_artifact,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
