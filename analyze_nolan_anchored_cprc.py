#!/usr/bin/env python3
"""Evaluate NoLan as an analytic anchor for paired CPRC calibration."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from analyze_cprc_gate_pareto import (
    DevelopmentSelector,
    apply_gate_family,
    family_gate_grid,
    joint_point,
    paired_counts,
    prepare_geometry,
    sha256,
)
from analyze_cprc_robustness import load_model_rows
from analyze_hierarchical_eb_lambda_cv import (
    candidate_arrays,
    dataset_metrics,
    fit_lambda_posterior,
    inverse_softplus,
    policy_details_gauss_hermite,
)


VERSION = "nolan_anchored_cprc_v2"


def nolan_lambda(row: Mapping[str, Any], beta: float = 0.8) -> float:
    _, image, prior = candidate_arrays(dict(row))
    image_prob = np.exp(image - np.max(image))
    image_prob /= image_prob.sum()
    prior_prob = np.exp(prior - np.max(prior))
    prior_prob /= prior_prob.sum()
    epsilon = 1e-12
    divergence = 0.5 * float(
        np.sum(
            image_prob
            * np.log(np.maximum(image_prob, epsilon) / np.maximum(prior_prob, epsilon))
        )
        + np.sum(
            prior_prob
            * np.log(np.maximum(prior_prob, epsilon) / np.maximum(image_prob, epsilon))
        )
    )
    alpha = float(beta) * (math.tanh(1.0 / max(divergence, epsilon)) + 1.0)
    return alpha / (1.0 + alpha)


def nolan_predictions(
    rows: Sequence[Dict[str, Any]], lambdas: np.ndarray
) -> list[str]:
    predictions = []
    for row, lambda_value in zip(rows, lambdas):
        keys, image, prior = candidate_arrays(row)
        predictions.append(keys[int(np.argmax(image - float(lambda_value) * prior))])
    return predictions


def raw_summary(
    rows: Sequence[Dict[str, Any]], geometry: Any, predictions: Sequence[str]
) -> Dict[str, Any]:
    development = dataset_metrics(rows, predictions, geometry.train_ids)
    test = dataset_metrics(rows, predictions, geometry.test_ids)
    return {
        "development_metrics": development,
        "development_joint": joint_point(development, ("dev_mc", "dev_qa")),
        "test_metrics": test,
        "test_joint": joint_point(test, ("test_mc", "test_qa")),
        "test_repairs_harms": paired_counts(rows, predictions, geometry.test_ids),
    }


def run_model(
    model: str, spec: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    rows = load_model_rows(spec)
    geometry = prepare_geometry(rows)
    released_beta = float(config["nolan_beta"])
    released_anchors = np.asarray(
        [nolan_lambda(row, released_beta) for row in rows], dtype=np.float64
    )
    raw_predictions = nolan_predictions(rows, released_anchors)

    selection = config["selection"]
    selector = DevelopmentSelector(
        rows,
        geometry,
        config["cs_budgets"],
        float(selection["cs_cost"]),
        bool(selection["require_nonnegative_cf_per_task"]),
    )
    quadrature = int(config["posterior"]["quadrature_points"])
    anchor_summaries = {}
    for beta_value in config.get("anchor_beta_grid", [released_beta]):
        beta = float(beta_value)
        anchors = np.asarray(
            [nolan_lambda(row, beta) for row in rows], dtype=np.float64
        )
        latent_offsets = np.asarray(
            [inverse_softplus(value) for value in anchors], dtype=np.float64
        )
        anchor_summaries[f"{beta:g}"] = {
            "minimum": float(np.min(anchors)),
            "q25": float(np.quantile(anchors, 0.25)),
            "median": float(np.median(anchors)),
            "q75": float(np.quantile(anchors, 0.75)),
            "maximum": float(np.max(anchors)),
        }

        fixed_details = policy_details_gauss_hermite(
            rows,
            geometry.contexts,
            np.zeros(geometry.contexts.shape[1], dtype=np.float64),
            np.zeros(
                (geometry.contexts.shape[1], geometry.contexts.shape[1]),
                dtype=np.float64,
            ),
            lambda_cap=0.8,
            quadrature_points=quadrature,
            support_distances=np.zeros(len(rows), dtype=np.float64),
            arity_supported=geometry.arity_supported,
            latent_offsets=latent_offsets,
        )
        for gate in family_gate_grid(
            "map_no_laplace_or_density", (float("inf"),)
        ):
            selector.consider(
                "nolan_support_gate",
                {"beta": beta, "gate": gate},
                apply_gate_family(
                    rows, fixed_details, gate, "map_no_laplace_or_density"
                ),
                fixed_details,
            )

        for l2_value in config["posterior"]["l2_grid"]:
            l2 = float(l2_value)
            theta, covariance, loss = fit_lambda_posterior(
                rows,
                geometry.contexts,
                geometry.train_ids,
                l2,
                latent_offsets=latent_offsets,
            )
            for cap_value in config["posterior"]["lambda_caps"]:
                cap = float(cap_value)
                post_details = policy_details_gauss_hermite(
                    rows,
                    geometry.contexts,
                    theta,
                    covariance,
                    lambda_cap=cap,
                    quadrature_points=quadrature,
                    support_distances=geometry.support_distances,
                    arity_supported=geometry.arity_supported,
                    latent_offsets=latent_offsets,
                )
                for gate in family_gate_grid(
                    "bprc_full", geometry.distance_thresholds
                ):
                    selector.consider(
                        "nolan_anchor_post",
                        {
                            "beta": beta,
                            "l2": l2,
                            "lambda_cap": cap,
                            "train_loss": loss,
                            "gate": gate,
                        },
                        apply_gate_family(rows, post_details, gate, "bprc_full"),
                        post_details,
                    )

                map_details = policy_details_gauss_hermite(
                    rows,
                    geometry.contexts,
                    theta,
                    np.zeros_like(covariance),
                    lambda_cap=cap,
                    quadrature_points=quadrature,
                    support_distances=np.zeros(len(rows), dtype=np.float64),
                    arity_supported=geometry.arity_supported,
                    latent_offsets=latent_offsets,
                )
                for gate in family_gate_grid(
                    "map_no_laplace_or_density", (float("inf"),)
                ):
                    selector.consider(
                        "nolan_anchor_map",
                        {
                            "beta": beta,
                            "l2": l2,
                            "lambda_cap": cap,
                            "train_loss": loss,
                            "gate": gate,
                        },
                        apply_gate_family(
                            rows, map_details, gate, "map_no_laplace_or_density"
                        ),
                        map_details,
                    )

    return {
        "model": model,
        "rows": len(rows),
        "development_rows": len(geometry.train_ids),
        "test_rows": len(geometry.test_ids),
        "released_beta": released_beta,
        "anchor_summaries": anchor_summaries,
        "nolan_raw": raw_summary(rows, geometry, raw_predictions),
        "selected_by_development_cs_budget": selector.serialize(),
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    print("model\tfamily\tbudget\ttask\tdCF\tdCS")
    for model, experiment in payload["experiments"].items():
        raw = experiment["nolan_raw"]["test_metrics"]
        for task in ("mc", "qa"):
            values = raw[f"test_{task}"]
            print(
                f"{model}\tnolan_raw\treleased\t{task}"
                f"\t{values['counterfactual']['delta']:+.4f}"
                f"\t{values['commonsense']['delta']:+.4f}"
            )
        selected = experiment["selected_by_development_cs_budget"]
        for family in (
            "nolan_support_gate",
            "nolan_anchor_map",
            "nolan_anchor_post",
        ):
            for budget, result in selected.get(family, {}).items():
                for task in ("mc", "qa"):
                    values = result["test_metrics"][f"test_{task}"]
                    print(
                        f"{model}\t{family}\t{budget}\t{task}"
                        f"\t{values['counterfactual']['delta']:+.4f}"
                        f"\t{values['commonsense']['delta']:+.4f}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cprc_nolan_anchor_v1.json")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    requested = set(args.model or config["models"])
    unknown = requested - set(config["models"])
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    payload = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "selection_split": "development_only",
        "blind_test_refit": False,
        "fairness": config["fairness"],
        "experiments": {},
    }
    for model, spec in config["models"].items():
        if model in requested:
            payload["experiments"][model] = run_model(model, spec, config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
