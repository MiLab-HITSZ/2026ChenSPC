#!/usr/bin/env python3
"""Compose frozen learned SPC with fixed and support-gated NoLan fallbacks."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from analyze_spc_gate_pareto import (
    DevelopmentSelector,
    apply_gate_family,
    family_gate_grid,
    prepare_geometry,
    sha256,
)
from analyze_spc_robustness import load_model_rows
from analyze_hierarchical_eb_lambda_cv import (
    fit_lambda_posterior,
    inverse_softplus,
    policy_details_gauss_hermite,
    resolved_baseline_key,
)
from analyze_nolan_anchored_spc import nolan_lambda


VERSION = "nolan_support_backfill_v1"


def backfill_predictions(
    rows: Sequence[Dict[str, Any]],
    primary: Sequence[str],
    secondary: Sequence[str],
) -> list[str]:
    output = []
    for row, first, second in zip(rows, primary, secondary):
        baseline = resolved_baseline_key(row)
        output.append(str(first) if str(first) != baseline else str(second))
    return output


def reconstruct_primary(
    rows: Sequence[Dict[str, Any]],
    geometry: Any,
    params: Mapping[str, Any],
    family: str,
    quadrature: int,
) -> list[str]:
    theta, covariance, _ = fit_lambda_posterior(
        rows,
        geometry.contexts,
        geometry.train_ids,
        float(params["l2"]),
    )
    if family == "map_no_laplace_or_density":
        covariance = np.zeros_like(covariance)
        support_distances = np.zeros(len(rows), dtype=np.float64)
    elif family == "bprc_full":
        support_distances = geometry.support_distances
    else:
        raise ValueError(f"unsupported primary family: {family}")
    details = policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        theta,
        covariance,
        lambda_cap=float(params["lambda_cap"]),
        quadrature_points=quadrature,
        support_distances=support_distances,
        arity_supported=geometry.arity_supported,
    )
    return apply_gate_family(rows, details, params["gate"], family)


def run_model(
    model: str,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    spc_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = load_model_rows(spec)
    geometry = prepare_geometry(rows)
    budget = f"{float(config['target_cs_budget']):.2f}"
    primary_family = str(config["primary_family"])
    spc_result = spc_artifact["experiments"][model][
        "selected_by_development_cs_budget"
    ][primary_family][budget]
    spc_predictions = reconstruct_primary(
        rows,
        geometry,
        spc_result["params"],
        family=primary_family,
        quadrature=int(config["quadrature_points"]),
    )

    beta = float(config["nolan_beta"])
    anchors = np.asarray([nolan_lambda(row, beta) for row in rows], dtype=np.float64)
    latent_offsets = np.asarray(
        [inverse_softplus(value) for value in anchors], dtype=np.float64
    )
    context_dimension = geometry.contexts.shape[1]
    nolan_details = policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        np.zeros(context_dimension, dtype=np.float64),
        np.zeros((context_dimension, context_dimension), dtype=np.float64),
        lambda_cap=0.8,
        quadrature_points=int(config["quadrature_points"]),
        support_distances=np.zeros(len(rows), dtype=np.float64),
        arity_supported=geometry.arity_supported,
        latent_offsets=latent_offsets,
    )
    raw_nolan = [str(detail["proposal"]) for detail in nolan_details]
    raw_union = backfill_predictions(rows, spc_predictions, raw_nolan)

    selection = config["selection"]
    selector = DevelopmentSelector(
        rows,
        geometry,
        [float(config["target_cs_budget"])],
        float(selection["cs_cost"]),
        bool(selection["require_nonnegative_cf_per_task"]),
    )
    for gate in family_gate_grid("map_no_laplace_or_density", (float("inf"),)):
        gated_nolan = apply_gate_family(
            rows, nolan_details, gate, "map_no_laplace_or_density"
        )
        selector.consider(
            "nolan_support_backfill",
            {
                "beta": beta,
                "primary": primary_family,
                "arbitration": "retain SPC override; otherwise use gated NoLan proposal",
                "gate": gate,
            },
            backfill_predictions(rows, spc_predictions, gated_nolan),
            nolan_details,
        )
    selected = selector.serialize()["nolan_support_backfill"][budget]
    raw_selector = DevelopmentSelector(
        rows,
        geometry,
        [1.0],
        float(selection["cs_cost"]),
        bool(selection["require_nonnegative_cf_per_task"]),
    )
    raw_selector.consider(
        "nolan_raw_backfill",
        {"beta": beta, "gate": "none"},
        raw_union,
        nolan_details,
    )
    raw_selected = raw_selector.serialize()["nolan_raw_backfill"]["1.00"]
    return {
        "model": model,
        "target_cs_budget": float(config["target_cs_budget"]),
        "primary_family": primary_family,
        "frozen_spc": spc_result,
        "nolan_raw_backfill": raw_selected,
        "nolan_support_backfill": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spc_nolan_backfill_v1.json")
    parser.add_argument(
        "--spc-artifact",
        default="result/cprc_robustness/gate_pareto_qwen_llava_v1.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    spc_path = Path(args.spc_artifact)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spc = json.loads(spc_path.read_text(encoding="utf-8"))
    payload = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "spc_artifact": str(spc_path),
        "spc_artifact_sha256": sha256(spc_path),
        "selection_split": "development_only",
        "test_refit": False,
        "models": {},
    }
    for model, spec in config["models"].items():
        payload["models"][model] = run_model(model, spec, config, spc)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("model\tmethod\ttask\tdCF\tdCS")
    for model, result in payload["models"].items():
        for method in ("frozen_spc", "nolan_raw_backfill", "nolan_support_backfill"):
            for task in ("mc", "qa"):
                values = result[method]["test_metrics"][f"test_{task}"]
                print(
                    f"{model}\t{method}\t{task}\t"
                    f"{values['counterfactual']['delta']:+.4f}\t"
                    f"{values['commonsense']['delta']:+.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
