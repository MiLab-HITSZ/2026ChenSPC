#!/usr/bin/env python3
"""Test whether paired CF/CS calibration is necessary for CPRC retention."""

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_cprc_gate_pareto import (
    apply_gate_family,
    family_gate_grid,
    json_safe,
)
from analyze_hierarchical_eb_lambda_cv import (
    context_features,
    exact_mcnemar_pvalue,
    fit_lambda_posterior,
    policy_details_gauss_hermite,
    resolved_baseline_key,
    standardize_contexts,
)


VERSION = "cprc_paired_calibration_v1"
TASKS = ("mc", "qa")
SIDES = ("counterfactual", "commonsense")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_development(row: Mapping[str, Any]) -> bool:
    return str(row.get("_dataset", "")).startswith("dev_")


def is_test(row: Mapping[str, Any]) -> bool:
    return str(row.get("_dataset", "")).startswith("test_")


def selected_ids(
    rows: Sequence[Mapping[str, Any]], split: str, side: Optional[str] = None
) -> List[int]:
    predicate = is_development if split == "development" else is_test
    return [
        index
        for index, row in enumerate(rows)
        if predicate(row) and (side is None or row.get("side") == side)
    ]


def endpoint_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    ids: Iterable[int],
    tasks: Sequence[str] = TASKS,
) -> Dict[str, Any]:
    id_list = list(ids)
    output: Dict[str, Any] = {}
    for task in tasks:
        output[task] = {}
        for side in SIDES:
            chosen = [
                index
                for index in id_list
                if rows[index].get("_task") == task and rows[index].get("side") == side
            ]
            baseline_correct = [
                normalize_key(resolved_baseline_key(dict(rows[index])))
                == normalize_key(rows[index].get("gt"))
                for index in chosen
            ]
            method_correct = [
                normalize_key(predictions[index])
                == normalize_key(rows[index].get("gt"))
                for index in chosen
            ]
            repairs = sum(
                not before and after
                for before, after in zip(baseline_correct, method_correct)
            )
            harms = sum(
                before and not after
                for before, after in zip(baseline_correct, method_correct)
            )
            count = len(chosen)
            output[task][side] = {
                "n": count,
                "baseline_correct": sum(baseline_correct),
                "method_correct": sum(method_correct),
                "baseline_acc": sum(baseline_correct) / count if count else None,
                "method_acc": sum(method_correct) / count if count else None,
                "delta": (
                    (sum(method_correct) - sum(baseline_correct)) / count
                    if count
                    else None
                ),
                "repairs": repairs,
                "harms": harms,
                "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(repairs, harms),
                "overrides": sum(
                    normalize_key(predictions[index])
                    != normalize_key(resolved_baseline_key(dict(rows[index])))
                    for index in chosen
                ),
            }
    return output


def mean_delta(
    metrics: Mapping[str, Any], side: str, tasks: Sequence[str] = TASKS
) -> float:
    return float(np.mean([metrics[task][side]["delta"] for task in tasks]))


@dataclass
class SelectedPolicy:
    key: Tuple[Any, ...]
    params_key: str
    params: Dict[str, Any]
    predictions: List[str]
    development_metrics: Dict[str, Any]


class VariantSelector:
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        variant: str,
        cs_budget: float,
        cs_cost: float,
        require_nonnegative_cf: bool,
        tasks: Sequence[str] = TASKS,
        selection_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.rows = rows
        self.variant = variant
        self.cs_budget = float(cs_budget)
        self.cs_cost = float(cs_cost)
        self.require_nonnegative_cf = bool(require_nonnegative_cf)
        self.tasks = tuple(tasks)
        self.dev_ids = list(
            selection_ids
            if selection_ids is not None
            else selected_ids(rows, "development")
        )
        self.best: Optional[SelectedPolicy] = None

    def _eligible(self, metrics: Mapping[str, Any]) -> bool:
        if self.require_nonnegative_cf and any(
            float(metrics[task]["counterfactual"]["delta"]) < -1e-12
            for task in self.tasks
        ):
            return False
        if self.variant != "paired_calibration":
            return True
        return all(
            float(metrics[task]["commonsense"]["delta"])
            >= -abs(self.cs_budget) - 1e-12
            for task in self.tasks
        )

    def _selection_key(
        self, metrics: Mapping[str, Any], predictions: Sequence[str]
    ) -> Tuple[Any, ...]:
        delta_cf = mean_delta(metrics, "counterfactual", self.tasks)
        delta_cs = mean_delta(metrics, "commonsense", self.tasks)
        overrides = sum(
            normalize_key(predictions[index])
            != normalize_key(resolved_baseline_key(dict(self.rows[index])))
            for index in self.dev_ids
        )
        if self.variant == "paired_calibration":
            return (
                delta_cf + self.cs_cost * delta_cs,
                delta_cf,
                delta_cs,
                -overrides,
            )
        # CS outcomes are deliberately absent from both selection and tie-breaking.
        return (delta_cf, -overrides)

    def consider(
        self,
        params: Mapping[str, Any],
        predictions: Sequence[str],
    ) -> None:
        metrics = endpoint_metrics(self.rows, predictions, self.dev_ids, self.tasks)
        if not self._eligible(metrics):
            return
        key = self._selection_key(metrics, predictions)
        params_key = json.dumps(json_safe(dict(params)), sort_keys=True)
        if self.best is None or key > self.best.key or (
            key == self.best.key and params_key < self.best.params_key
        ):
            self.best = SelectedPolicy(
                key=key,
                params_key=params_key,
                params=dict(params),
                predictions=list(predictions),
                development_metrics=metrics,
            )

    def serialize(self, test_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        if self.best is None:
            raise RuntimeError(f"no policy selected for {self.variant}")
        evaluation_ids = list(
            test_ids if test_ids is not None else selected_ids(self.rows, "test")
        )
        return {
            "params": json_safe(self.best.params),
            "development_metrics": self.best.development_metrics,
            "test_metrics": endpoint_metrics(
                self.rows, self.best.predictions, evaluation_ids, self.tasks
            ),
            "selection_key": list(self.best.key),
            "test_predictions": self.best.predictions,
        }


def fit_regime(
    rows: Sequence[Dict[str, Any]],
    fit_ids: Sequence[int],
    selectors: Sequence[VariantSelector],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    raw = np.stack([context_features(row) for row in rows])
    contexts, mean, scale = standardize_contexts(raw, fit_ids)
    arity_min = float(np.min(raw[list(fit_ids), 1]))
    arity_max = float(np.max(raw[list(fit_ids), 1]))
    arity_supported = np.logical_and(
        raw[:, 1] >= arity_min - 1e-12,
        raw[:, 1] <= arity_max + 1e-12,
    )
    point_map = config["point_map"]
    for l2_value in point_map["l2_grid"]:
        l2 = float(l2_value)
        theta, covariance, loss = fit_lambda_posterior(
            rows, contexts, fit_ids, l2
        )
        for cap_value in point_map["lambda_caps"]:
            cap = float(cap_value)
            details = policy_details_gauss_hermite(
                rows,
                contexts,
                theta,
                np.zeros_like(covariance),
                lambda_cap=cap,
                quadrature_points=int(point_map["quadrature_points"]),
                support_distances=np.zeros(len(rows)),
                arity_supported=arity_supported,
            )
            for gate in family_gate_grid(
                "map_no_laplace_or_density", (float("inf"),)
            ):
                predictions = apply_gate_family(
                    rows, details, gate, "map_no_laplace_or_density"
                )
                params = {
                    "l2": l2,
                    "lambda_cap": cap,
                    "gate": gate,
                    "train_loss": loss,
                }
                for selector in selectors:
                    selector.consider(params, predictions)
    return {
        "fit_rows": len(fit_ids),
        "context_mean": mean.tolist(),
        "context_scale": scale.tolist(),
        "arity_support": [arity_min, arity_max],
    }


def run_model(
    model: str, spec: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    tasks = tuple(str(task) for task in spec.get("tasks", TASKS))
    rows: List[Dict[str, Any]] = []
    seen = set()
    for task in tasks:
        paths = (
            (f"dev_{task}", spec[f"dev_{task}"]),
            (f"test_{task}", spec.get(f"test_{task}", spec.get("test"))),
        )
        for label, path in paths:
            if not path:
                raise ValueError(f"missing {label} path for {model}")
            for source_row in read_jsonl(Path(path), task):
                row = copy.deepcopy(source_row)
                key = (task, str(row.get("pair_id")), str(row.get("side")))
                if key in seen:
                    raise ValueError(f"duplicate model row: {key}")
                seen.add(key)
                row["_dataset"] = label
                row["_task"] = task
                rows.append(row)
    dev_all = selected_ids(rows, "development")
    dev_cf = selected_ids(rows, "development", "counterfactual")
    common = {
        "rows": rows,
        "cs_budget": float(config["paired_cs_budget"]),
        "cs_cost": float(config["paired_cs_cost"]),
        "require_nonnegative_cf": bool(config["require_nonnegative_cf_per_task"]),
        "tasks": tasks,
    }
    paired = VariantSelector(variant="paired_calibration", **common)
    cf_objective = VariantSelector(variant="paired_fit_cf_objective", **common)
    paired_fit = fit_regime(rows, dev_all, (paired, cf_objective), config)

    cf_only = VariantSelector(variant="cf_only_data", **common)
    cf_fit = fit_regime(rows, dev_cf, (cf_only,), config)

    variants = {
        "paired_calibration": paired.serialize(),
        "cf_only_data": cf_only.serialize(),
        "paired_fit_cf_objective": cf_objective.serialize(),
    }
    for value in variants.values():
        value.pop("test_predictions", None)
    return {
        "model": model,
        "tasks": list(tasks),
        "development_pairs": len(
            {rows[index].get("pair_id") for index in dev_all}
        ),
        "test_pairs": len(
            {rows[index].get("pair_id") for index in selected_ids(rows, "test")}
        ),
        "paired_fit": paired_fit,
        "cf_only_fit": cf_fit,
        "variants": variants,
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    print("model\tvariant\ttask\tdCF\tdCS\tCF_R/H\tCS_R/H")
    for model, experiment in payload["experiments"].items():
        for variant, result in experiment["variants"].items():
            for task in experiment.get("tasks", TASKS):
                metrics = result["test_metrics"][task]
                cf = metrics["counterfactual"]
                cs = metrics["commonsense"]
                print(
                    f"{model}\t{variant}\t{task}"
                    f"\t{cf['delta']:+.4f}\t{cs['delta']:+.4f}"
                    f"\t{cf['repairs']}/{cf['harms']}"
                    f"\t{cs['repairs']}/{cs['harms']}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/cprc_paired_calibration_v1.json"
    )
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
        "test_refit": False,
        "experiments": {
            model: run_model(model, spec, config)
            for model, spec in config["models"].items()
            if model in requested
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
