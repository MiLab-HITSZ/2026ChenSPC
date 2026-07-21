#!/usr/bin/env python3
"""Development-selected gate ablations and matched-CS operating points for BPRC."""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from analyze_spc_robustness import load_model_rows
from analyze_hierarchical_eb_lambda_cv import (
    candidate_arrays,
    context_features,
    dataset_metrics,
    fit_lambda_posterior,
    gate_grid,
    policy_details_gauss_hermite,
    resolved_baseline_key,
    selection_key,
    standardize_contexts,
    valid_retention,
)


VERSION = "bprc_gate_pareto_v1"
SIDES = ("counterfactual", "commonsense")


@dataclass
class Geometry:
    raw_contexts: np.ndarray
    contexts: np.ndarray
    support_distances: np.ndarray
    distance_thresholds: List[float]
    arity_supported: np.ndarray
    train_ids: List[int]
    test_ids: List[int]
    context_mean: np.ndarray
    context_scale: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def prepare_geometry(
    rows: Sequence[Dict[str, Any]], feature_mode: str = "full"
) -> Geometry:
    train_ids = [
        index
        for index, row in enumerate(rows)
        if str(row.get("_dataset", "")).startswith("dev_")
    ]
    test_ids = [
        index
        for index, row in enumerate(rows)
        if str(row.get("_dataset", "")).startswith("test_")
    ]
    raw = np.stack(
        [context_features(row, feature_mode=feature_mode) for row in rows]
    )
    contexts, mean, scale = standardize_contexts(raw, train_ids)
    density = contexts[:, 1:]
    train_density = density[train_ids]
    covariance = np.cov(train_density, rowvar=False)
    precision = np.linalg.pinv(
        np.atleast_2d(covariance) + 0.1 * np.eye(density.shape[1]), rcond=1e-8
    )
    distances = np.sqrt(
        np.maximum(0.0, np.einsum("ni,ij,nj->n", density, precision, density))
    )
    distance_thresholds = sorted(
        {
            float(np.quantile(distances[train_ids], quantile))
            for quantile in (0.9, 0.95, 0.99)
        }
    )
    arity_min = float(np.min(raw[train_ids, 1]))
    arity_max = float(np.max(raw[train_ids, 1]))
    arity_supported = np.logical_and(
        raw[:, 1] >= arity_min - 1e-12,
        raw[:, 1] <= arity_max + 1e-12,
    )
    return Geometry(
        raw_contexts=raw,
        contexts=contexts,
        support_distances=distances,
        distance_thresholds=distance_thresholds,
        arity_supported=arity_supported,
        train_ids=train_ids,
        test_ids=test_ids,
        context_mean=mean,
        context_scale=scale,
    )


def _gate_passes(
    detail: Mapping[str, Any], params: Mapping[str, float], family: str
) -> bool:
    if detail["proposal"] == detail["baseline"]:
        return False
    conflict = bool(detail["visual_conflict"] or detail["prior_absorption"])
    arity = bool(detail.get("arity_supported", True))
    density = float(detail.get("support_distance", 0.0)) <= float(
        params.get("support_distance_max", float("inf"))
    )
    margin = float(detail["posterior_margin"]) >= float(
        params.get("posterior_margin_min", 0.0)
    )
    support_threshold = (
        float(params.get("visual_support_min", 0.0))
        if detail["visual_conflict"]
        else float(params.get("absorption_support_min", 0.0))
    )
    posterior = float(detail["posterior_support"]) >= support_threshold and margin

    if family == "bprc_full":
        return arity and density and conflict and posterior
    if family == "bprc_no_density_gate":
        return arity and conflict and posterior
    if family == "bprc_no_posterior_gate":
        return arity and density and conflict
    if family == "bprc_no_conflict_gate":
        return arity and density and posterior
    if family == "bprc_conflict_only":
        return conflict
    if family == "bprc_proposal_only":
        return True
    if family == "map_no_laplace_or_density":
        return arity and conflict and posterior
    if family == "fixed_lambda_same_gate":
        return arity and density and conflict and posterior
    raise ValueError(f"unknown gate family: {family}")


def apply_gate_family(
    rows: Sequence[Dict[str, Any]],
    details: Sequence[Mapping[str, Any]],
    params: Mapping[str, float],
    family: str,
) -> List[str]:
    return [
        str(detail["proposal"])
        if _gate_passes(detail, params, family)
        else resolved_baseline_key(row)
        for row, detail in zip(rows, details)
    ]


def family_gate_grid(family: str, distances: Sequence[float]) -> List[Dict[str, float]]:
    if family in {"bprc_full", "fixed_lambda_same_gate"}:
        return gate_grid(distances)
    if family in {"bprc_no_density_gate", "map_no_laplace_or_density"}:
        return gate_grid((float("inf"),))
    if family == "bprc_no_posterior_gate":
        return [
            {
                "visual_support_min": 0.0,
                "absorption_support_min": 0.0,
                "posterior_margin_min": 0.0,
                "support_distance_max": float(distance),
            }
            for distance in distances
        ]
    if family == "bprc_no_conflict_gate":
        return gate_grid(distances)
    if family in {"bprc_conflict_only", "bprc_proposal_only"}:
        return [
            {
                "visual_support_min": 0.0,
                "absorption_support_min": 0.0,
                "posterior_margin_min": 0.0,
                "support_distance_max": float("inf"),
            }
        ]
    raise ValueError(f"unknown gate family: {family}")


def joint_point(metrics: Mapping[str, Any], labels: Sequence[str]) -> Dict[str, float]:
    cf = [float(metrics[label]["counterfactual"]["delta"]) for label in labels]
    cs = [float(metrics[label]["commonsense"]["delta"]) for label in labels]
    return {
        "delta_cf": float(np.mean(cf)),
        "delta_cs": float(np.mean(cs)),
        "utility_0_5": float(np.mean(cf) + 0.5 * np.mean(cs)),
    }


def paired_counts(
    rows: Sequence[Dict[str, Any]], predictions: Sequence[str], ids: Iterable[int]
) -> Dict[str, Dict[str, int]]:
    output: Dict[str, Dict[str, int]] = {}
    for label in sorted({str(rows[index]["_dataset"]) for index in ids}):
        output[label] = {}
        label_ids = [index for index in ids if rows[index]["_dataset"] == label]
        for side in SIDES:
            selected = [index for index in label_ids if rows[index].get("side") == side]
            repairs = harms = 0
            for index in selected:
                gt = str(rows[index].get("gt", "")).strip().lower()
                baseline_correct = resolved_baseline_key(rows[index]).lower() == gt
                method_correct = str(predictions[index]).lower() == gt
                repairs += int(not baseline_correct and method_correct)
                harms += int(baseline_correct and not method_correct)
            output[label][side] = {"repairs": repairs, "harms": harms}
    return output


class DevelopmentSelector:
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        geometry: Geometry,
        budgets: Sequence[float],
        cs_cost: float,
        require_cf_nonnegative: bool,
    ) -> None:
        self.rows = rows
        self.geometry = geometry
        self.budgets = [float(value) for value in budgets]
        self.cs_cost = float(cs_cost)
        self.require_cf_nonnegative = bool(require_cf_nonnegative)
        self.best: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.frontiers: Dict[str, List[Dict[str, Any]]] = {}

    def _update_frontier(
        self, family: str, params: Mapping[str, Any], metrics: Mapping[str, Any]
    ) -> None:
        labels = ("dev_mc", "dev_qa")
        point = {**joint_point(metrics, labels), "params": dict(params)}
        frontier = self.frontiers.setdefault(family, [])
        if any(
            other["delta_cf"] >= point["delta_cf"] - 1e-12
            and other["delta_cs"] >= point["delta_cs"] - 1e-12
            and (
                other["delta_cf"] > point["delta_cf"] + 1e-12
                or other["delta_cs"] > point["delta_cs"] + 1e-12
            )
            for other in frontier
        ):
            return
        frontier[:] = [
            other
            for other in frontier
            if not (
                point["delta_cf"] >= other["delta_cf"] - 1e-12
                and point["delta_cs"] >= other["delta_cs"] - 1e-12
                and (
                    point["delta_cf"] > other["delta_cf"] + 1e-12
                    or point["delta_cs"] > other["delta_cs"] + 1e-12
                )
            )
        ]
        if not any(
            abs(other["delta_cf"] - point["delta_cf"]) < 1e-12
            and abs(other["delta_cs"] - point["delta_cs"]) < 1e-12
            for other in frontier
        ):
            frontier.append(point)

    def consider(
        self,
        family: str,
        params: Mapping[str, Any],
        predictions: Sequence[str],
        details: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        metrics = dataset_metrics(self.rows, predictions, self.geometry.train_ids)
        self._update_frontier(family, params, metrics)
        primary = selection_key(
            metrics,
            predictions,
            self.rows,
            self.geometry.train_ids,
            selection_objective="net_utility",
            cs_cost=self.cs_cost,
        )
        param_key = json.dumps(params, sort_keys=True, allow_nan=True)
        for budget in self.budgets:
            if not valid_retention(metrics, budget, self.require_cf_nonnegative):
                continue
            budget_key = f"{budget:.2f}"
            current = self.best.setdefault(family, {}).get(budget_key)
            better = current is None or primary > current["selection_key"]
            if current is not None and primary == current["selection_key"]:
                better = param_key < current["param_key"]
            if better:
                self.best[family][budget_key] = {
                    "params": dict(params),
                    "predictions": list(predictions),
                    "details": details,
                    "development_metrics": metrics,
                    "selection_key": primary,
                    "param_key": param_key,
                }

    def serialize(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for family, by_budget in sorted(self.best.items()):
            output[family] = {}
            for budget, candidate in sorted(by_budget.items(), key=lambda item: float(item[0])):
                predictions = candidate["predictions"]
                test_metrics = dataset_metrics(
                    self.rows, predictions, self.geometry.test_ids
                )
                output[family][budget] = {
                    "params": json_safe(candidate["params"]),
                    "development_metrics": candidate["development_metrics"],
                    "development_joint": joint_point(
                        candidate["development_metrics"], ("dev_mc", "dev_qa")
                    ),
                    "test_metrics": test_metrics,
                    "test_joint": joint_point(test_metrics, ("test_mc", "test_qa")),
                    "test_repairs_harms": paired_counts(
                        self.rows, predictions, self.geometry.test_ids
                    ),
                }
        return output


def fit_override_logistic(
    contexts: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    proposals: Sequence[str],
    train_ids: Sequence[int],
    l2: float,
) -> np.ndarray:
    x = torch.as_tensor(contexts[list(train_ids)], dtype=torch.float64)
    targets = []
    for index in train_ids:
        gt = str(rows[index].get("gt", "")).strip().lower()
        proposal_gain = int(str(proposals[index]).lower() == gt) - int(
            resolved_baseline_key(rows[index]).lower() == gt
        )
        targets.append(float(proposal_gain > 0))
    y = torch.as_tensor(targets, dtype=torch.float64)
    theta = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    positive_rate = float(y.mean())
    if 0.0 < positive_rate < 1.0:
        theta.data[0] = math.log(positive_rate / (1.0 - positive_rate))
    optimizer = torch.optim.LBFGS(
        [theta], max_iter=120, tolerance_grad=1e-9, tolerance_change=1e-11
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x @ theta
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="sum"
        ) + 0.5 * float(l2) * (theta @ theta)
        loss.backward()
        return loss

    optimizer.step(closure)
    return theta.detach().numpy()


def threshold_grid(probabilities: np.ndarray, train_ids: Sequence[int]) -> List[float]:
    unique = sorted({float(probabilities[index]) for index in train_ids})
    if not unique:
        return [1.01]
    thresholds = [0.0, 1.01]
    thresholds.extend(unique)
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    return sorted(set(thresholds))


def gated_predictions(
    rows: Sequence[Dict[str, Any]],
    proposals: Sequence[str],
    probabilities: np.ndarray,
    threshold: float,
) -> List[str]:
    return [
        str(proposal)
        if str(proposal) != resolved_baseline_key(row)
        and float(probability) >= float(threshold)
        else resolved_baseline_key(row)
        for row, proposal, probability in zip(rows, proposals, probabilities)
    ]


def corrected_proposals(
    rows: Sequence[Dict[str, Any]], coefficient: float, beta: float = 0.0
) -> List[str]:
    output = []
    log_beta = math.log(min(float(beta), 1.0)) if float(beta) > 0.0 else None
    for row in rows:
        keys, image, prior = candidate_arrays(row)
        eligible = np.ones(len(keys), dtype=bool)
        if log_beta is not None:
            eligible = image >= float(np.max(image)) + log_beta
            if not np.any(eligible):
                eligible[:] = True
        scores = image - float(coefficient) * prior
        scores = np.where(eligible, scores, float("-inf"))
        output.append(keys[int(np.argmax(scores))])
    return output


def run_posterior_families(
    rows: Sequence[Dict[str, Any]],
    geometry: Geometry,
    selector: DevelopmentSelector,
    posterior_config: Mapping[str, Any],
) -> None:
    quadrature = int(posterior_config["quadrature_points"])
    full_families = (
        "bprc_full",
        "bprc_no_density_gate",
        "bprc_no_posterior_gate",
        "bprc_no_conflict_gate",
        "bprc_conflict_only",
        "bprc_proposal_only",
    )
    fitted: Dict[float, Tuple[np.ndarray, np.ndarray, float]] = {}
    for l2_value in posterior_config["l2_grid"]:
        l2 = float(l2_value)
        theta, covariance, loss = fit_lambda_posterior(
            rows, geometry.contexts, geometry.train_ids, l2
        )
        fitted[l2] = (theta, covariance, loss)
        for cap_value in posterior_config["lambda_caps"]:
            cap = float(cap_value)
            details = policy_details_gauss_hermite(
                rows,
                geometry.contexts,
                theta,
                covariance,
                lambda_cap=cap,
                quadrature_points=quadrature,
                support_distances=geometry.support_distances,
                arity_supported=geometry.arity_supported,
            )
            for family in full_families:
                for gate in family_gate_grid(family, geometry.distance_thresholds):
                    predictions = apply_gate_family(rows, details, gate, family)
                    selector.consider(
                        family,
                        {"l2": l2, "lambda_cap": cap, "gate": gate},
                        predictions,
                        details,
                    )

            map_details = policy_details_gauss_hermite(
                rows,
                geometry.contexts,
                theta,
                np.zeros_like(covariance),
                lambda_cap=cap,
                quadrature_points=quadrature,
                support_distances=np.zeros(len(rows)),
                arity_supported=geometry.arity_supported,
            )
            family = "map_no_laplace_or_density"
            for gate in family_gate_grid(family, (float("inf"),)):
                selector.consider(
                    family,
                    {"l2": l2, "lambda_cap": cap, "gate": gate},
                    apply_gate_family(rows, map_details, gate, family),
                    map_details,
                )

    constant_contexts = np.ones((len(rows), 1), dtype=np.float64)
    for l2_value in posterior_config["l2_grid"]:
        l2 = float(l2_value)
        theta, covariance, _ = fit_lambda_posterior(
            rows, constant_contexts, geometry.train_ids, l2
        )
        for cap_value in posterior_config["lambda_caps"]:
            cap = float(cap_value)
            details = policy_details_gauss_hermite(
                rows,
                constant_contexts,
                theta,
                covariance,
                lambda_cap=cap,
                quadrature_points=quadrature,
                support_distances=geometry.support_distances,
                arity_supported=geometry.arity_supported,
            )
            family = "fixed_lambda_same_gate"
            for gate in family_gate_grid(family, geometry.distance_thresholds):
                selector.consider(
                    family,
                    {"l2": l2, "lambda_cap": cap, "gate": gate},
                    apply_gate_family(rows, details, gate, family),
                    details,
                )


def run_direct_gate_controls(
    rows: Sequence[Dict[str, Any]],
    geometry: Geometry,
    selector: DevelopmentSelector,
    config: Mapping[str, Any],
) -> None:
    proposal_sets = []
    for gamma_value in config["pai_gammas"]:
        gamma = float(gamma_value)
        coefficient = (gamma - 1.0) / gamma
        proposals = corrected_proposals(rows, coefficient, float(config["pai_beta"]))
        proposal_sets.append(("pai_adapted_13d_gate", {"gamma": gamma}, proposals))
        selector.consider(
            "pai_adapted_ungated",
            {"gamma": gamma, "beta": float(config["pai_beta"])},
            [
                proposal if proposal != resolved_baseline_key(row) else resolved_baseline_key(row)
                for row, proposal in zip(rows, proposals)
            ],
        )
    for coefficient_value in config["fixed_lambdas"]:
        coefficient = float(coefficient_value)
        proposal_sets.append(
            (
                "fixed_lambda_13d_gate",
                {"lambda": coefficient},
                corrected_proposals(rows, coefficient),
            )
        )

    for family, proposal_params, proposals in proposal_sets:
        for l2_value in config["l2_grid"]:
            l2 = float(l2_value)
            theta = fit_override_logistic(
                geometry.contexts, rows, proposals, geometry.train_ids, l2
            )
            logits = geometry.contexts @ theta
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            for threshold in threshold_grid(probabilities, geometry.train_ids):
                params = {
                    **proposal_params,
                    "gate_l2": l2,
                    "threshold": float(threshold),
                    "gate_theta": theta.tolist(),
                }
                selector.consider(
                    family,
                    params,
                    gated_predictions(rows, proposals, probabilities, threshold),
                )


def frozen_deletions(
    rows: Sequence[Dict[str, Any]],
    geometry: Geometry,
    selected: Mapping[str, Any],
) -> Dict[str, Any]:
    primary = selected.get("bprc_full", {}).get("0.10")
    if primary is None:
        return {}
    params = primary["params"]
    theta, covariance, _ = fit_lambda_posterior(
        rows, geometry.contexts, geometry.train_ids, float(params["l2"])
    )
    details = policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        theta,
        covariance,
        lambda_cap=float(params["lambda_cap"]),
        quadrature_points=64,
        support_distances=geometry.support_distances,
        arity_supported=geometry.arity_supported,
    )
    gate = params["gate"]
    families = (
        "bprc_full",
        "bprc_no_density_gate",
        "bprc_no_posterior_gate",
        "bprc_no_conflict_gate",
        "bprc_conflict_only",
        "bprc_proposal_only",
    )
    output = {}
    for family in families:
        predictions = apply_gate_family(rows, details, gate, family)
        output[family] = {
            "test_metrics": dataset_metrics(rows, predictions, geometry.test_ids),
            "test_repairs_harms": paired_counts(rows, predictions, geometry.test_ids),
        }
    for name, predicate in (
        ("visual_conflict_only", lambda detail: bool(detail["visual_conflict"])),
        ("prior_absorption_only", lambda detail: bool(detail["prior_absorption"])),
    ):
        predictions = [
            str(detail["proposal"])
            if predicate(detail) and _gate_passes(detail, gate, "bprc_full")
            else resolved_baseline_key(row)
            for row, detail in zip(rows, details)
        ]
        output[name] = {
            "test_metrics": dataset_metrics(rows, predictions, geometry.test_ids),
            "test_repairs_harms": paired_counts(rows, predictions, geometry.test_ids),
        }
    return {
        "frozen_from": "bprc_full development CS budget 0.10",
        "params": params,
        "variants": output,
    }


def run_model(
    model: str, spec: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    rows = load_model_rows(spec)
    geometry = prepare_geometry(rows)
    selection = config["selection"]
    selector = DevelopmentSelector(
        rows,
        geometry,
        config["cs_budgets"],
        float(selection["cs_cost"]),
        bool(selection["require_nonnegative_cf_per_task"]),
    )
    run_posterior_families(rows, geometry, selector, config["posterior"])
    run_direct_gate_controls(rows, geometry, selector, config["direct_gate"])
    selected = selector.serialize()
    return {
        "model": model,
        "rows": len(rows),
        "development_rows": len(geometry.train_ids),
        "test_rows": len(geometry.test_ids),
        "context_mean": geometry.context_mean.tolist(),
        "context_scale": geometry.context_scale.tolist(),
        "distance_thresholds": geometry.distance_thresholds,
        "selected_by_development_cs_budget": selected,
        "development_pareto_frontiers": {
            family: json_safe(
                sorted(
                    points, key=lambda point: (point["delta_cs"], point["delta_cf"])
                )
            )
            for family, points in selector.frontiers.items()
        },
        "frozen_gate_deletions": frozen_deletions(rows, geometry, selected),
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    print("model\tfamily\tbudget\ttask\tdCF\tdCS")
    for model, experiment in payload["experiments"].items():
        selected = experiment["selected_by_development_cs_budget"]
        for family in (
            "bprc_full",
            "fixed_lambda_same_gate",
            "map_no_laplace_or_density",
            "pai_adapted_ungated",
            "pai_adapted_13d_gate",
            "fixed_lambda_13d_gate",
        ):
            for budget, result in selected.get(family, {}).items():
                for task in ("mc", "qa"):
                    metrics = result["test_metrics"][f"test_{task}"]
                    print(
                        f"{model}\t{family}\t{budget}\t{task}"
                        f"\t{metrics['counterfactual']['delta']:+.4f}"
                        f"\t{metrics['commonsense']['delta']:+.4f}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spc_gate_pareto_v1.json")
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
        "cs_budgets": config["cs_budgets"],
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
