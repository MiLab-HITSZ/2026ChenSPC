#!/usr/bin/env python3
"""Audit candidate lengths and compare summed versus mean token log scores."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from analyze_cp_vbc_bayes_path_cv import normalize_key
from analyze_spc_paired_calibration import (
    VariantSelector,
    endpoint_metrics,
    fit_regime,
    selected_ids,
)
from analyze_spc_robustness import load_model_rows
from analyze_spc_gate_pareto import json_safe
from analyze_hierarchical_eb_lambda_cv import candidate_arrays


VERSION = "candidate_score_normalization_v1"
TASKS = ("mc", "qa")
SIDES = ("counterfactual", "commonsense")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_score_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = copy.deepcopy(list(rows))
    for row in output:
        for candidate in (row.get("cp_vbc") or {}).get("candidates") or []:
            candidate["logp_image"] = float(candidate["avg_logp_image"])
            candidate["logp_prior"] = float(candidate["avg_logp_prior"])
    return output


def length_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    token_counts: Counter = Counter()
    row_patterns: Counter = Counter()
    unequal_within_row = 0
    image_prior_length_mismatch = 0
    sum_mean_identity_mismatch = 0
    for row in rows:
        candidates = (row.get("cp_vbc") or {}).get("candidates") or []
        pattern = []
        for candidate in candidates:
            image_count = int(candidate["image_token_count"])
            prior_count = int(candidate["prior_token_count"])
            token_counts[(str(row.get("_task")), image_count, prior_count)] += 1
            pattern.append((image_count, prior_count))
            image_prior_length_mismatch += int(image_count != prior_count)
            sum_mean_identity_mismatch += int(
                abs(
                    float(candidate["logp_image"])
                    - image_count * float(candidate["avg_logp_image"])
                )
                > 1e-5
            )
            sum_mean_identity_mismatch += int(
                abs(
                    float(candidate["logp_prior"])
                    - prior_count * float(candidate["avg_logp_prior"])
                )
                > 1e-5
            )
        row_patterns[(str(row.get("_task")), tuple(pattern))] += 1
        unequal_within_row += int(len(set(pattern)) > 1)
    return {
        "rows": len(rows),
        "candidate_token_counts": [
            {
                "task": task,
                "image_tokens": image_tokens,
                "prior_tokens": prior_tokens,
                "candidates": count,
            }
            for (task, image_tokens, prior_tokens), count in sorted(
                token_counts.items()
            )
        ],
        "row_patterns": [
            {
                "task": task,
                "candidate_lengths": [list(value) for value in pattern],
                "rows": count,
            }
            for (task, pattern), count in sorted(row_patterns.items())
        ],
        "rows_with_unequal_candidate_lengths": unequal_within_row,
        "image_prior_length_mismatches": image_prior_length_mismatch,
        "sum_mean_identity_mismatches": sum_mean_identity_mismatch,
    }


def fixed_residual_agreement(
    sum_rows: Sequence[Dict[str, Any]],
    mean_rows: Sequence[Dict[str, Any]],
    lambdas: Sequence[float],
) -> Dict[str, Any]:
    disagree = 0
    total = 0
    by_lambda = {}
    for coefficient in lambdas:
        local_disagree = 0
        for sum_row, mean_row in zip(sum_rows, mean_rows):
            sum_keys, sum_image, sum_prior = candidate_arrays(sum_row)
            mean_keys, mean_image, mean_prior = candidate_arrays(mean_row)
            if sum_keys != mean_keys:
                raise ValueError("candidate key mismatch between score normalizations")
            sum_top = sum_keys[int(np.argmax(sum_image - coefficient * sum_prior))]
            mean_top = mean_keys[int(np.argmax(mean_image - coefficient * mean_prior))]
            local_disagree += int(sum_top != mean_top)
        by_lambda[f"{float(coefficient):.2f}"] = {
            "rows": len(sum_rows),
            "disagreements": local_disagree,
            "agreement": 1.0 - local_disagree / len(sum_rows),
        }
        disagree += local_disagree
        total += len(sum_rows)
    return {
        "comparisons": total,
        "disagreements": disagree,
        "agreement": 1.0 - disagree / total,
        "by_lambda": by_lambda,
    }


def fit_paired_policy(
    rows: Sequence[Dict[str, Any]], paired_config: Mapping[str, Any]
) -> VariantSelector:
    selector = VariantSelector(
        rows=rows,
        variant="paired_calibration",
        cs_budget=float(paired_config["paired_cs_budget"]),
        cs_cost=float(paired_config["paired_cs_cost"]),
        require_nonnegative_cf=bool(
            paired_config["require_nonnegative_cf_per_task"]
        ),
    )
    fit_regime(
        rows,
        selected_ids(rows, "development"),
        (selector,),
        paired_config,
    )
    if selector.best is None:
        raise RuntimeError("paired score-normalization policy was not selected")
    return selector


def prediction_agreement(
    rows: Sequence[Mapping[str, Any]],
    left: Sequence[str],
    right: Sequence[str],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for task in TASKS:
        output[task] = {}
        for side in SIDES:
            ids = [
                index
                for index in selected_ids(rows, "test")
                if rows[index].get("_task") == task and rows[index].get("side") == side
            ]
            matches = sum(
                normalize_key(left[index]) == normalize_key(right[index])
                for index in ids
            )
            output[task][side] = {
                "n": len(ids),
                "matches": matches,
                "agreement": matches / len(ids),
            }
    return output


def run_model(
    model: str,
    spec: Mapping[str, Any],
    paired_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    sum_rows = load_model_rows(spec)
    mean_rows = mean_score_rows(sum_rows)
    sum_policy = fit_paired_policy(sum_rows, paired_config)
    mean_policy = fit_paired_policy(mean_rows, paired_config)
    test_ids = selected_ids(sum_rows, "test")
    return {
        "model": model,
        "primary_normalization": "sum",
        "length_audit": length_audit(sum_rows),
        "fixed_residual_sum_mean": fixed_residual_agreement(
            sum_rows, mean_rows, config["fixed_lambda_grid"]
        ),
        "selected": {
            "sum": {
                "params": json_safe(sum_policy.best.params),
                "test_metrics": endpoint_metrics(
                    sum_rows, sum_policy.best.predictions, test_ids
                ),
            },
            "mean": {
                "params": json_safe(mean_policy.best.params),
                "test_metrics": endpoint_metrics(
                    mean_rows, mean_policy.best.predictions, test_ids
                ),
            },
        },
        "selected_prediction_agreement": prediction_agreement(
            sum_rows, sum_policy.best.predictions, mean_policy.best.predictions
        ),
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    print("model\tmode\ttask\tdCF\tdCS")
    for model, experiment in payload["experiments"].items():
        for mode, selected in experiment["selected"].items():
            for task in TASKS:
                metrics = selected["test_metrics"][task]
                print(
                    f"{model}\t{mode}\t{task}"
                    f"\t{metrics['counterfactual']['delta']:+.4f}"
                    f"\t{metrics['commonsense']['delta']:+.4f}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/candidate_score_normalization_v1.json"
    )
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paired_path = Path(config["paired_calibration_config"])
    paired_config = json.loads(paired_path.read_text(encoding="utf-8"))
    requested = set(args.model or paired_config["models"])
    unknown = requested - set(paired_config["models"])
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    payload = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "paired_calibration_config": str(paired_path),
        "paired_calibration_config_sha256": sha256(paired_path),
        "test_refit": False,
        "experiments": {
            model: run_model(model, spec, paired_config, config)
            for model, spec in paired_config["models"].items()
            if model in requested
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
