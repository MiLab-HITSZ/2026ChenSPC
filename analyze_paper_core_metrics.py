#!/usr/bin/env python3
"""Recompute the compact metric and attribution tables used in the paper."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from analyze_spc_gate_pareto import (
    apply_gate_family,
    corrected_proposals,
    prepare_geometry,
)
from analyze_spc_robustness import load_model_rows
from analyze_hierarchical_eb_lambda_cv import (
    candidate_arrays,
    fit_lambda_posterior,
    policy_details_gauss_hermite,
    resolved_baseline_key,
)


VERSION = "paper_core_metrics_v1"
PRIMARY_FAMILY = "map_no_laplace_or_density"


def _test_ids(rows: Sequence[Mapping[str, Any]], task: str) -> list[int]:
    label = f"test_{task}"
    return [
        index
        for index, row in enumerate(rows)
        if row.get("_dataset") == label
    ]


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    ids: Iterable[int],
) -> Dict[str, Any]:
    selected = list(ids)
    by_side: Dict[str, Dict[str, Any]] = {}
    for side in ("counterfactual", "commonsense"):
        side_ids = [index for index in selected if rows[index].get("side") == side]
        correct = sum(
            str(predictions[index]).strip().lower()
            == str(rows[index].get("gt", "")).strip().lower()
            for index in side_ids
        )
        by_side[side] = {
            "n": len(side_ids),
            "correct": correct,
            "accuracy": correct / len(side_ids) if side_ids else None,
        }

    cf = by_side["counterfactual"]["accuracy"]
    cs = by_side["commonsense"]["accuracy"]
    cfad = cs - cf
    cf_error_ids = [
        index
        for index in selected
        if rows[index].get("side") == "counterfactual"
        and str(predictions[index]).strip().lower()
        != str(rows[index].get("gt", "")).strip().lower()
    ]
    commonsense_errors = sum(
        str(predictions[index]).strip().lower()
        == str(rows[index].get("cs_gt", "")).strip().lower()
        for index in cf_error_ids
    )
    return {
        **by_side,
        "CFAD": cfad,
        "CCR": commonsense_errors / len(cf_error_ids) if cf_error_ids else None,
        "RPD": cfad / cs if cs else None,
        "cf_errors": len(cf_error_ids),
        "cf_commonsense_errors": commonsense_errors,
    }


def _predictions_for_selected(
    rows: Sequence[Dict[str, Any]],
    geometry: Any,
    selected: Mapping[str, Any],
    quadrature_points: int,
) -> tuple[list[str], list[Mapping[str, Any]]]:
    params = selected["params"]
    theta, covariance, _ = fit_lambda_posterior(
        rows,
        geometry.contexts,
        geometry.train_ids,
        float(params["l2"]),
    )
    details = policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        theta,
        np.zeros_like(covariance),
        lambda_cap=float(params["lambda_cap"]),
        quadrature_points=quadrature_points,
        support_distances=np.zeros(len(rows)),
        arity_supported=geometry.arity_supported,
    )
    predictions = apply_gate_family(
        rows,
        details,
        params["gate"],
        PRIMARY_FAMILY,
    )
    return predictions, details


def _proposal_predictions(
    rows: Sequence[Mapping[str, Any]],
    details: Sequence[Mapping[str, Any]],
) -> list[str]:
    return [
        str(detail["proposal"])
        if str(detail["proposal"]) != resolved_baseline_key(row)
        else resolved_baseline_key(row)
        for row, detail in zip(rows, details)
    ]


def _agreement(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    ids: Iterable[int],
) -> Dict[str, Any]:
    selected = list(ids)
    native = [resolved_baseline_key(rows[index]) for index in selected]
    image = []
    prior = []
    for index in selected:
        keys, image_scores, prior_scores = candidate_arrays(rows[index])
        image.append(keys[int(np.argmax(image_scores))])
        prior.append(keys[int(np.argmax(prior_scores))])

    native_image_agreement = sum(
        left == right for left, right in zip(native, image)
    )
    method_repairs = []
    method_harms = []
    for local_index, row_index in enumerate(selected):
        gt = str(rows[row_index].get("gt", "")).strip().lower()
        native_correct = native[local_index].lower() == gt
        method_correct = str(predictions[row_index]).lower() == gt
        if not native_correct and method_correct:
            method_repairs.append(local_index)
        if native_correct and not method_correct:
            method_harms.append(local_index)

    repair_image_already_correct = sum(
        image[index].lower()
        == str(rows[selected[index]].get("gt", "")).strip().lower()
        for index in method_repairs
    )
    repair_all_winners_agree = sum(
        native[index] == image[index] == prior[index]
        for index in method_repairs
    )
    return {
        "n": len(selected),
        "native_image_agreement": native_image_agreement,
        "native_image_agreement_rate": (
            native_image_agreement / len(selected) if selected else None
        ),
        "spc_repairs": len(method_repairs),
        "spc_harms": len(method_harms),
        "repairs_where_image_scorer_was_already_correct": (
            repair_image_already_correct
        ),
        "repairs_where_image_scorer_was_already_correct_rate": (
            repair_image_already_correct / len(method_repairs)
            if method_repairs
            else None
        ),
        "repairs_where_native_image_and_prior_winners_agreed": (
            repair_all_winners_agree
        ),
        "repairs_where_native_image_and_prior_winners_agreed_rate": (
            repair_all_winners_agree / len(method_repairs)
            if method_repairs
            else None
        ),
    }


def _fixed_subtraction_metrics(
    rows: Sequence[Dict[str, Any]],
    coefficient: float,
    task: str,
) -> Dict[str, Any]:
    predictions = corrected_proposals(rows, coefficient)
    return _metrics(rows, predictions, _test_ids(rows, task))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/spc_gate_pareto_v1.json",
    )
    parser.add_argument(
        "--selection",
        default="result/cprc_robustness/gate_pareto_qwen_llava_v1.json",
    )
    parser.add_argument(
        "--output",
        default="result/paper_revision_stats/paper_core_metrics.json",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    quadrature_points = int(config["posterior"]["quadrature_points"])
    payload: Dict[str, Any] = {
        "version": VERSION,
        "config": args.config,
        "selection": args.selection,
        "primary_family": PRIMARY_FAMILY,
        "models": {},
    }

    for model, spec in config["models"].items():
        rows = load_model_rows(spec)
        geometry = prepare_geometry(rows)
        model_selection = selection["experiments"][model][
            "selected_by_development_cs_budget"
        ][PRIMARY_FAMILY]
        model_output: Dict[str, Any] = {"tasks": {}, "fixed_subtraction": {}}
        for budget in ("0.04", "0.10"):
            predictions, details = _predictions_for_selected(
                rows,
                geometry,
                model_selection[budget],
                quadrature_points,
            )
            proposal_predictions = _proposal_predictions(rows, details)
            budget_output = {}
            for task in ("mc", "qa"):
                ids = _test_ids(rows, task)
                native_predictions = [
                    resolved_baseline_key(row) for row in rows
                ]
                image_predictions = []
                for row in rows:
                    keys, image_scores, _ = candidate_arrays(row)
                    image_predictions.append(keys[int(np.argmax(image_scores))])
                budget_output[task] = {
                    "native": _metrics(rows, native_predictions, ids),
                    "image_scorer": _metrics(rows, image_predictions, ids),
                    "learned_correction_without_gate": _metrics(
                        rows,
                        proposal_predictions,
                        ids,
                    ),
                    "spc": _metrics(rows, predictions, ids),
                    "agreement": {
                        side: _agreement(
                            rows,
                            predictions,
                            [
                                index
                                for index in ids
                                if rows[index].get("side") == side
                            ],
                        )
                        for side in ("counterfactual", "commonsense")
                    },
                }
            model_output["tasks"][budget] = budget_output

        for coefficient in (0.3, 0.5, 0.8, 1.0):
            model_output["fixed_subtraction"][f"{coefficient:.1f}"] = {
                task: _fixed_subtraction_metrics(rows, coefficient, task)
                for task in ("mc", "qa")
            }
        payload["models"][model] = model_output

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("model\tbudget\ttask\tmethod\tCF\tCS\tCFAD\tCCR\tRPD")
    for model, model_output in payload["models"].items():
        for budget, by_task in model_output["tasks"].items():
            for task, result in by_task.items():
                for method in ("native", "image_scorer", "spc"):
                    metrics = result[method]
                    print(
                        f"{model}\t{budget}\t{task}\t{method}"
                        f"\t{metrics['counterfactual']['accuracy']:.4f}"
                        f"\t{metrics['commonsense']['accuracy']:.4f}"
                        f"\t{metrics['CFAD']:.4f}"
                        f"\t{metrics['CCR']:.4f}"
                        f"\t{metrics['RPD']:.4f}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
