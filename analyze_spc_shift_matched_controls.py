#!/usr/bin/env python3
"""Matched-capacity BPRC controls under semantic and calibration shift."""

import argparse
import copy
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from analyze_spc_calibration_size import development_pair_order, selected_manifest
from analyze_spc_gate_pareto import (
    DevelopmentSelector,
    apply_gate_family,
    corrected_proposals,
    family_gate_grid,
    fit_override_logistic,
    gated_predictions,
    json_safe,
    paired_counts,
    prepare_geometry,
    threshold_grid,
)
from analyze_spc_robustness import load_model_rows
from analyze_hierarchical_eb_lambda_cv import (
    exact_mcnemar_pvalue,
    fit_lambda_posterior,
    policy_details_gauss_hermite,
    resolved_baseline_key,
)


VERSION = "bprc_shift_matched_controls_v1"
CORE_FAMILIES = (
    "bprc_full",
    "map_no_laplace_or_density",
    "fixed_lambda_same_gate",
    "fixed_lambda_13d_gate",
)
TASKS = ("mc", "qa")
SIDES = ("counterfactual", "commonsense")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relabel_rows(
    rows: Sequence[Mapping[str, Any]],
    train_pair_ids: Iterable[str],
    test_predicate: Any,
) -> List[Dict[str, Any]]:
    """Keep score rows fixed while making train/test membership explicit."""
    train_pairs = {str(value) for value in train_pair_ids}
    output: List[Dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        task = str(row.get("_task"))
        dataset = str(row.get("_dataset"))
        pair_id = str(row.get("pair_id"))
        if dataset.startswith("dev_") and pair_id in train_pairs:
            row["_dataset"] = f"dev_{task}"
        elif dataset.startswith("test_") and bool(test_predicate(row)):
            row["_dataset"] = f"test_{task}"
        else:
            row["_dataset"] = f"ignore_{task}"
        output.append(row)
    return output


def development_pairs(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted(
        {
            str(row.get("pair_id"))
            for row in rows
            if str(row.get("_dataset", "")).startswith("dev_")
        }
    )


def run_core_families(
    rows: Sequence[Dict[str, Any]], config: Mapping[str, Any]
) -> Tuple[DevelopmentSelector, Any]:
    geometry = prepare_geometry(rows)
    selection = config["selection"]
    selector = DevelopmentSelector(
        rows,
        geometry,
        config["cs_budgets"],
        float(selection["cs_cost"]),
        bool(selection["require_nonnegative_cf_per_task"]),
    )
    posterior = config["posterior"]
    quadrature = int(posterior["quadrature_points"])

    for l2_value in posterior["l2_grid"]:
        l2 = float(l2_value)
        theta, covariance, _ = fit_lambda_posterior(
            rows, geometry.contexts, geometry.train_ids, l2
        )
        for cap_value in posterior["lambda_caps"]:
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
            for gate in family_gate_grid(
                "bprc_full", geometry.distance_thresholds
            ):
                selector.consider(
                    "bprc_full",
                    {"l2": l2, "lambda_cap": cap, "gate": gate},
                    apply_gate_family(rows, details, gate, "bprc_full"),
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
            for gate in family_gate_grid(
                "map_no_laplace_or_density", (float("inf"),)
            ):
                selector.consider(
                    "map_no_laplace_or_density",
                    {"l2": l2, "lambda_cap": cap, "gate": gate},
                    apply_gate_family(
                        rows, map_details, gate, "map_no_laplace_or_density"
                    ),
                    map_details,
                )

    constant_contexts = np.ones((len(rows), 1), dtype=np.float64)
    for l2_value in posterior["l2_grid"]:
        l2 = float(l2_value)
        theta, covariance, _ = fit_lambda_posterior(
            rows, constant_contexts, geometry.train_ids, l2
        )
        for cap_value in posterior["lambda_caps"]:
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
            for gate in family_gate_grid(
                "fixed_lambda_same_gate", geometry.distance_thresholds
            ):
                selector.consider(
                    "fixed_lambda_same_gate",
                    {"l2": l2, "lambda_cap": cap, "gate": gate},
                    apply_gate_family(rows, details, gate, "fixed_lambda_same_gate"),
                    details,
                )

    direct = config["direct_gate"]
    for coefficient_value in direct["fixed_lambdas"]:
        coefficient = float(coefficient_value)
        proposals = corrected_proposals(rows, coefficient)
        for l2_value in direct["l2_grid"]:
            l2 = float(l2_value)
            theta = fit_override_logistic(
                geometry.contexts, rows, proposals, geometry.train_ids, l2
            )
            logits = geometry.contexts @ theta
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            for threshold in threshold_grid(probabilities, geometry.train_ids):
                params = {
                    "lambda": coefficient,
                    "gate_l2": l2,
                    "threshold": float(threshold),
                    "gate_theta": theta.tolist(),
                }
                selector.consider(
                    "fixed_lambda_13d_gate",
                    params,
                    gated_predictions(rows, proposals, probabilities, threshold),
                )
    return selector, geometry


def compact_selected(
    rows: Sequence[Dict[str, Any]],
    selector: DevelopmentSelector,
    geometry: Any,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[Tuple[str, str, str], bool]]]]:
    serialized: Dict[str, Any] = {}
    outcomes: Dict[str, Dict[str, Dict[Tuple[str, str, str], bool]]] = {}
    for family in CORE_FAMILIES:
        serialized[family] = {}
        outcomes[family] = {}
        for budget, candidate in sorted(
            selector.best.get(family, {}).items(), key=lambda item: float(item[0])
        ):
            predictions = candidate["predictions"]
            test_metrics = {}
            for task in TASKS:
                ids = [
                    index
                    for index in geometry.test_ids
                    if rows[index].get("_task") == task
                ]
                from analyze_hierarchical_eb_lambda_cv import dataset_metrics

                task_metrics = dataset_metrics(rows, predictions, ids)
                test_metrics.update(task_metrics)
            serialized[family][budget] = {
                "params": json_safe(candidate["params"]),
                "development_metrics": candidate["development_metrics"],
                "test_metrics": test_metrics,
                "test_repairs_harms": paired_counts(
                    rows, predictions, geometry.test_ids
                ),
            }
            outcomes[family][budget] = {}
            for index in geometry.test_ids:
                row = rows[index]
                key = (
                    str(row.get("_task")),
                    str(row.get("pair_id")),
                    str(row.get("side")),
                )
                outcomes[family][budget][key] = (
                    str(predictions[index]).strip().lower()
                    == str(row.get("gt", "")).strip().lower()
                )
    return serialized, outcomes


def aggregate_metric_blocks(blocks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    totals: MutableMapping[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {
            "n": 0,
            "baseline_correct": 0,
            "correct": 0,
            "overrides": 0,
        }
    )
    for block in blocks:
        for label, by_side in block.items():
            task = str(label).removeprefix("test_")
            for side, values in by_side.items():
                target = totals[(task, side)]
                for field in target:
                    target[field] += int(values[field])
    output: Dict[str, Any] = {}
    for task in TASKS:
        output[task] = {}
        for side in SIDES:
            values = totals[(task, side)]
            n = values["n"]
            output[task][side] = {
                **values,
                "baseline_acc": values["baseline_correct"] / n if n else 0.0,
                "acc": values["correct"] / n if n else 0.0,
                "delta": (
                    (values["correct"] - values["baseline_correct"]) / n
                    if n
                    else 0.0
                ),
            }
    return output


def utility(metrics: Mapping[str, Any], cs_cost: float) -> float:
    delta_cf = statistics.mean(
        float(metrics[task]["counterfactual"]["delta"]) for task in TASKS
    )
    delta_cs = statistics.mean(
        float(metrics[task]["commonsense"]["delta"]) for task in TASKS
    )
    return delta_cf + float(cs_cost) * delta_cs


def compare_outcomes(
    reference: Mapping[Tuple[str, str, str], bool],
    candidate: Mapping[Tuple[str, str, str], bool],
) -> Dict[str, Any]:
    keys = sorted(set(reference) & set(candidate))
    candidate_repairs = sum(not reference[key] and candidate[key] for key in keys)
    candidate_harms = sum(reference[key] and not candidate[key] for key in keys)
    return {
        "n": len(keys),
        "candidate_minus_reference_accuracy": (
            (candidate_repairs - candidate_harms) / len(keys) if keys else 0.0
        ),
        "candidate_repairs_over_reference": candidate_repairs,
        "candidate_harms_vs_reference": candidate_harms,
        "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(
            candidate_repairs, candidate_harms
        ),
    }


def aggregate_fold_results(
    folds: Sequence[Mapping[str, Any]], budgets: Sequence[float], cs_cost: float
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for budget_value in budgets:
        budget = f"{float(budget_value):.2f}"
        output[budget] = {"families": {}, "head_to_head_vs_full": {}}
        merged_outcomes: Dict[str, Dict[Tuple[str, str, str], bool]] = {}
        for family in CORE_FAMILIES:
            selected = [
                fold["selected"][family][budget]["test_metrics"] for fold in folds
            ]
            metrics = aggregate_metric_blocks(selected)
            output[budget]["families"][family] = {
                "metrics": metrics,
                "utility": utility(metrics, cs_cost),
            }
            merged: Dict[Tuple[str, str, str], bool] = {}
            for fold in folds:
                merged.update(fold["outcomes"][family][budget])
            merged_outcomes[family] = merged
        reference = merged_outcomes["bprc_full"]
        for family in CORE_FAMILIES[1:]:
            output[budget]["head_to_head_vs_full"][family] = compare_outcomes(
                reference, merged_outcomes[family]
            )
    return output


def leave_supercategory_out(
    rows: Sequence[Dict[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    groups = sorted(
        {
            str(row.get("category"))
            for row in rows
            if str(row.get("_dataset", "")).startswith("test_")
        }
    )
    all_development = development_pairs(rows)
    folds = []
    for group in groups:
        allowed_pairs = {
            str(row.get("pair_id"))
            for row in rows
            if str(row.get("_dataset", "")).startswith("dev_")
            and str(row.get("category")) != group
        }
        shifted = relabel_rows(
            rows,
            allowed_pairs,
            lambda row, heldout=group: str(row.get("category")) == heldout,
        )
        selector, geometry = run_core_families(shifted, config)
        selected, outcomes = compact_selected(shifted, selector, geometry)
        folds.append(
            {
                "heldout_supercategory": group,
                "calibration_pairs": len(allowed_pairs),
                "excluded_development_pairs": len(set(all_development) - allowed_pairs),
                "test_rows": len(geometry.test_ids),
                "selected": selected,
                "outcomes": outcomes,
            }
        )
    aggregate = aggregate_fold_results(
        folds, config["cs_budgets"], float(config["selection"]["cs_cost"])
    )
    serializable_folds = copy.deepcopy(folds)
    for fold in serializable_folds:
        fold.pop("outcomes", None)
    return {
        "folds": serializable_folds,
        "aggregate": aggregate,
        "guarantee": (
            "Every evaluated row belongs to a complete semantic supercategory absent "
            "from calibration; group metadata is never an inference feature."
        ),
    }


def calibration_scarcity(
    rows: Sequence[Dict[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    trials = []
    for seed in config["shift"]["calibration_subset_seeds"]:
        order = development_pair_order(rows, int(seed))
        manifest = selected_manifest(
            order, int(config["shift"]["pairs_per_subcategory"])
        )
        selected_pairs = {
            pair_id for pair_ids in manifest.values() for pair_id in pair_ids
        }
        shifted = relabel_rows(rows, selected_pairs, lambda _row: True)
        selector, geometry = run_core_families(shifted, config)
        selected, outcomes = compact_selected(shifted, selector, geometry)
        trials.append(
            {
                "seed": int(seed),
                "calibration_pairs": len(selected_pairs),
                "manifest": manifest,
                "selected": selected,
                "outcomes": outcomes,
            }
        )

    summary: Dict[str, Any] = {}
    for budget_value in config["cs_budgets"]:
        budget = f"{float(budget_value):.2f}"
        summary[budget] = {}
        for family in CORE_FAMILIES:
            values = []
            for trial in trials:
                metrics = aggregate_metric_blocks(
                    [trial["selected"][family][budget]["test_metrics"]]
                )
                values.append(
                    {
                        "metrics": metrics,
                        "utility": utility(
                            metrics, float(config["selection"]["cs_cost"])
                        ),
                    }
                )
            summary[budget][family] = {
                "trials": values,
                "mean_utility": statistics.mean(value["utility"] for value in values),
                "std_utility": statistics.pstdev(
                    value["utility"] for value in values
                ),
                "mean_delta": {
                    task: {
                        side: statistics.mean(
                            value["metrics"][task][side]["delta"] for value in values
                        )
                        for side in SIDES
                    }
                    for task in TASKS
                },
            }
    serializable_trials = copy.deepcopy(trials)
    for trial in serializable_trials:
        trial.pop("outcomes", None)
    return {
        "trials": serializable_trials,
        "summary": summary,
        "guarantee": (
            "Each trial uses the same one pair per subcategory for both image sides "
            "and answer formats; the 230-pair target never participates in selection."
        ),
    }


def method_position(experiments: Mapping[str, Any], budget: str) -> Dict[str, Any]:
    evidence = []
    for model, result in experiments.items():
        leaveout = result["leave_supercategory_out"]["aggregate"][budget]["families"]
        scarcity = result["calibration_scarcity"]["summary"][budget]
        for shift_name, values in (
            ("leave_supercategory_out", leaveout),
            ("calibration_scarcity", scarcity),
        ):
            full = (
                float(values["bprc_full"]["utility"])
                if shift_name == "leave_supercategory_out"
                else float(values["bprc_full"]["mean_utility"])
            )
            control_utilities = {
                family: (
                    float(values[family]["utility"])
                    if shift_name == "leave_supercategory_out"
                    else float(values[family]["mean_utility"])
                )
                for family in CORE_FAMILIES[1:]
            }
            point = control_utilities["map_no_laplace_or_density"]
            strongest_family, strongest = max(
                control_utilities.items(), key=lambda item: item[1]
            )
            evidence.append(
                {
                    "model": model,
                    "shift": shift_name,
                    "full_minus_point_map_utility": full - point,
                    "strongest_control": strongest_family,
                    "full_minus_strongest_control_utility": full - strongest,
                }
            )
    advantages = [
        item["full_minus_strongest_control_utility"] for item in evidence
    ]
    posterior_core_supported = bool(advantages) and min(advantages) >= 0.005
    return {
        "budget": budget,
        "evidence": evidence,
        "posterior_core_supported": posterior_core_supported,
        "recommended_core": (
            "posterior risk control"
            if posterior_core_supported
            else "candidate-aligned prior residual"
        ),
        "decision_rule": (
            "Posterior risk control is promoted to the algorithmic core only if it "
            "beats every matched-capacity control by at least 0.5 utility points "
            "under both shifts for both checkpoints; otherwise it remains an "
            "optional retention layer."
        ),
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    print("model\tshift\tbudget\tfamily\ttask\tdCF\tdCS\tutility")
    for model, experiment in payload["experiments"].items():
        for budget, result in experiment["leave_supercategory_out"]["aggregate"].items():
            for family, values in result["families"].items():
                for task in TASKS:
                    metrics = values["metrics"][task]
                    print(
                        f"{model}\tleave-supercategory\t{budget}\t{family}\t{task}"
                        f"\t{metrics['counterfactual']['delta']:+.4f}"
                        f"\t{metrics['commonsense']['delta']:+.4f}"
                        f"\t{values['utility']:+.4f}"
                    )
        for budget, result in experiment["calibration_scarcity"]["summary"].items():
            for family, values in result.items():
                for task in TASKS:
                    delta = values["mean_delta"][task]
                    print(
                        f"{model}\tcalibration-scarcity\t{budget}\t{family}\t{task}"
                        f"\t{delta['counterfactual']:+.4f}"
                        f"\t{delta['commonsense']:+.4f}"
                        f"\t{values['mean_utility']:+.4f}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/spc_shift_matched_controls_v1.json"
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

    payload: Dict[str, Any] = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "families": list(CORE_FAMILIES),
        "selection_split": "development_only within every shifted fold",
        "experiments": {},
    }
    for model, spec in config["models"].items():
        if model not in requested:
            continue
        rows = load_model_rows(spec)
        payload["experiments"][model] = {
            "leave_supercategory_out": leave_supercategory_out(rows, config),
            "calibration_scarcity": calibration_scarcity(rows, config),
        }
    decision_budget = f"{float(config['shift']['decision_budget']):.2f}"
    payload["method_position"] = method_position(
        payload["experiments"], decision_budget
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print_summary(payload)
    print(json.dumps(payload["method_position"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
