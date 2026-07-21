#!/usr/bin/env python3
"""Replay one frozen BPRC fit across logically equivalent QA templates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_candidate_baseline_sweep import _method_block, _replay_pai
from analyze_hierarchical_eb_lambda_cv import (
    apply_gate,
    context_features,
    fit_lambda_posterior,
    policy_details_gauss_hermite,
    resolved_baseline_key,
    standardize_contexts,
)


VERSION = "qa_template_stress_analysis_v2"
SIDES = ("counterfactual", "commonsense")
VARIANTS = ("cs_first", "cf_first")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260715


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: str, dataset: str) -> List[Dict[str, Any]]:
    output = []
    for source in read_jsonl(Path(path), "qa"):
        row = dict(source)
        row["_dataset"] = dataset
        row["_task"] = "qa"
        output.append(row)
    return output


def prepare_rows(spec: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[int]]:
    rows = []
    rows.extend(load_rows(spec["train_cs_first"], "train_contrastive_cs_first"))
    rows.extend(load_rows(spec["train_cf_first"], "train_contrastive_cf_first"))
    train_ids = list(range(len(rows)))
    for template_id, variants in spec["templates"].items():
        for variant, path in variants.items():
            rows.extend(load_rows(path, f"test_{template_id}_{variant}"))
    return rows, train_ids


def fit_frozen_policy(
    rows: Sequence[Dict[str, Any]],
    train_ids: Sequence[int],
    selected_params: Mapping[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    raw = np.stack([context_features(row) for row in rows])
    contexts, mean, scale = standardize_contexts(raw, train_ids)
    density = contexts[:, 1:]
    train_density = density[list(train_ids)]
    covariance = np.cov(train_density, rowvar=False)
    precision = np.linalg.pinv(
        np.atleast_2d(covariance) + 0.1 * np.eye(density.shape[1]), rcond=1e-8
    )
    distances = np.sqrt(
        np.maximum(0.0, np.einsum("ni,ij,nj->n", density, precision, density))
    )
    arity_min = float(np.min(raw[list(train_ids), 1]))
    arity_max = float(np.max(raw[list(train_ids), 1]))
    arity_supported = np.logical_and(
        raw[:, 1] >= arity_min - 1e-12,
        raw[:, 1] <= arity_max + 1e-12,
    )
    theta, posterior_covariance, loss = fit_lambda_posterior(
        rows, contexts, train_ids, float(selected_params["l2"])
    )
    stored_theta = np.asarray(selected_params.get("theta") or theta)
    max_theta_error = float(np.max(np.abs(theta - stored_theta)))
    if max_theta_error > 1e-7:
        raise ValueError(f"frozen theta replay mismatch: {max_theta_error}")
    details = policy_details_gauss_hermite(
        rows,
        contexts,
        theta,
        posterior_covariance,
        lambda_cap=float(selected_params["lambda_cap"]),
        quadrature_points=64,
        support_distances=distances,
        arity_supported=arity_supported,
    )
    predictions = apply_gate(rows, details, selected_params["gate"])
    return predictions, {
        "l2": float(selected_params["l2"]),
        "lambda_cap": float(selected_params["lambda_cap"]),
        "gate": selected_params["gate"],
        "train_loss": loss,
        "theta_replay_max_abs_error": max_theta_error,
        "context_mean": mean.tolist(),
        "context_scale": scale.tolist(),
    }


def correct(prediction: str, row: Mapping[str, Any]) -> bool:
    return normalize_key(prediction) == normalize_key(row.get("gt"))


def exact_mcnemar(repairs: int, harms: int) -> float:
    discordant = repairs + harms
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(repairs, harms) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (float(item[1]), item[0]))
    count = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in values}


def bootstrap_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def paired_bootstrap_ci(differences: Sequence[int], label: str) -> List[float]:
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed(label))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def side_metrics(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    dataset: str,
    side: str,
) -> Dict[str, Any]:
    ids = [
        index
        for index, row in enumerate(rows)
        if row["_dataset"] == dataset and row.get("side") == side
    ]
    baseline = [resolved_baseline_key(rows[index]) for index in ids]
    method = [predictions[index] for index in ids]
    baseline_correct = [correct(value, rows[index]) for value, index in zip(baseline, ids)]
    method_correct = [correct(value, rows[index]) for value, index in zip(method, ids)]
    n = len(ids)
    repairs = sum(not before and after for before, after in zip(baseline_correct, method_correct))
    harms = sum(before and not after for before, after in zip(baseline_correct, method_correct))
    differences = [
        int(after) - int(before)
        for before, after in zip(baseline_correct, method_correct)
    ]
    return {
        "n": n,
        "baseline_acc": sum(baseline_correct) / n,
        "method_acc": sum(method_correct) / n,
        "delta": (sum(method_correct) - sum(baseline_correct)) / n,
        "repairs": repairs,
        "harms": harms,
        "mcnemar_p": exact_mcnemar(repairs, harms),
        "delta_ci95": paired_bootstrap_ci(differences, f"{dataset}:{side}"),
        "baseline_yes_rate": sum(normalize_key(value) == "yes" for value in baseline) / n,
        "method_yes_rate": sum(normalize_key(value) == "yes" for value in method) / n,
    }


def row_lookup(
    rows: Sequence[Dict[str, Any]], dataset: str, side: str
) -> Dict[str, int]:
    output = {
        str(row["pair_id"]): index
        for index, row in enumerate(rows)
        if row["_dataset"] == dataset and row.get("side") == side
    }
    if len(output) != 230:
        raise ValueError(f"{dataset}/{side} has {len(output)} pairs")
    return output


def prior_top(row: Mapping[str, Any]) -> str:
    block = row.get("cp_vbc") or {}
    value = normalize_key(block.get("prior_top"))
    if value not in {"yes", "no"}:
        raise ValueError(f"invalid no-image prior top: {block.get('prior_top')!r}")
    return value


def paired_template_metrics(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    template_id: str,
    side: str,
) -> Dict[str, Any]:
    left = row_lookup(rows, f"test_{template_id}_cs_first", side)
    right = row_lookup(rows, f"test_{template_id}_cf_first", side)
    if set(left) != set(right):
        raise ValueError(f"pair mismatch for {template_id}/{side}")
    counters = {
        "baseline_complement": 0,
        "baseline_joint": 0,
        "method_complement": 0,
        "method_joint": 0,
        "prior_complement": 0,
    }
    for pair_id in sorted(left):
        left_id, right_id = left[pair_id], right[pair_id]
        baseline_pair = {
            normalize_key(resolved_baseline_key(rows[left_id])),
            normalize_key(resolved_baseline_key(rows[right_id])),
        }
        method_pair = {
            normalize_key(predictions[left_id]),
            normalize_key(predictions[right_id]),
        }
        prior_pair = {prior_top(rows[left_id]), prior_top(rows[right_id])}
        counters["baseline_complement"] += int(baseline_pair == {"yes", "no"})
        counters["method_complement"] += int(method_pair == {"yes", "no"})
        counters["prior_complement"] += int(prior_pair == {"yes", "no"})
        counters["baseline_joint"] += int(
            correct(resolved_baseline_key(rows[left_id]), rows[left_id])
            and correct(resolved_baseline_key(rows[right_id]), rows[right_id])
        )
        counters["method_joint"] += int(
            correct(predictions[left_id], rows[left_id])
            and correct(predictions[right_id], rows[right_id])
        )
    return {key: value / len(left) for key, value in counters.items()}


def summarize_templates(
    rows: Sequence[Dict[str, Any]], predictions: Sequence[str], template_ids: Sequence[str]
) -> Dict[str, Any]:
    output = {}
    for template_id in template_ids:
        output[template_id] = {
            "by_variant": {
                variant: {
                    side: side_metrics(
                        rows,
                        predictions,
                        f"test_{template_id}_{variant}",
                        side,
                    )
                    for side in SIDES
                }
                for variant in VARIANTS
            },
            "paired": {
                side: paired_template_metrics(
                    rows, predictions, template_id, side
                )
                for side in SIDES
            },
        }
    return output


def aggregate_range(per_template: Mapping[str, Any]) -> Dict[str, Any]:
    output = {}
    for variant in VARIANTS:
        output[variant] = {}
        for side in SIDES:
            values = [
                float(template["by_variant"][variant][side]["delta"])
                for template in per_template.values()
            ]
            output[variant][side] = {
                "mean_delta": float(np.mean(values)),
                "min_delta": min(values),
                "max_delta": max(values),
                "all_nonnegative": all(value >= -1e-12 for value in values),
            }
    return output


def pai_predictions(
    rows: Sequence[Dict[str, Any]], gamma: float
) -> List[str]:
    params = {
        "gamma": float(gamma),
        "beta": 0.1,
        "gate": "always",
        "contrast_margin": 0.0,
    }
    output = []
    for row in rows:
        block = _method_block(dict(row), "pai")
        if block is None:
            raise ValueError(
                f"missing image/no-image candidate block at {row.get('_dataset')}"
            )
        prediction = _replay_pai(block, params)
        if prediction is None:
            raise ValueError(f"PAI replay failed at {row.get('_dataset')}")
        output.append(normalize_key(prediction))
    return output


def mapping_prediction(policy: str, baseline: str) -> str:
    baseline = normalize_key(baseline)
    if policy == "identity":
        return baseline
    if policy == "always_yes":
        return "yes"
    if policy == "always_no":
        return "no"
    if policy == "invert":
        return "no" if baseline == "yes" else "yes"
    raise ValueError(f"unknown mapping policy: {policy}")


def mapping_development_metrics(
    rows: Sequence[Dict[str, Any]], predictions: Sequence[str]
) -> Dict[str, Any]:
    by_variant = {
        variant: {
            side: side_metrics(
                rows, predictions, f"train_contrastive_{variant}", side
            )
            for side in SIDES
        }
        for variant in VARIANTS
    }
    utility = sum(
        values["counterfactual"]["delta"]
        + 0.5 * values["commonsense"]["delta"]
        for values in by_variant.values()
    )
    eligible = all(
        values["counterfactual"]["delta"] >= -1e-12
        and values["commonsense"]["delta"] >= -0.2 - 1e-12
        for values in by_variant.values()
    )
    return {
        "eligible": eligible,
        "utility_0_5_sum": utility,
        "by_variant": by_variant,
    }


def direction_only_control(
    rows: Sequence[Dict[str, Any]], template_ids: Sequence[str]
) -> Dict[str, Any]:
    development = []
    candidates: Dict[str, List[str]] = {}
    for policy in ("identity", "always_yes", "always_no", "invert"):
        predictions = [
            mapping_prediction(policy, resolved_baseline_key(row)) for row in rows
        ]
        candidates[policy] = predictions
        development.append(
            {"policy": policy, **mapping_development_metrics(rows, predictions)}
        )
    selected = max(
        (item for item in development if item["eligible"]),
        key=lambda item: (
            item["utility_0_5_sum"],
            item["policy"] == "identity",
        ),
    )
    per_template = summarize_templates(
        rows, candidates[selected["policy"]], template_ids
    )
    return {
        "development": development,
        "selected_policy": selected["policy"],
        "per_template": per_template,
        "delta_ranges": aggregate_range(per_template),
    }


def run_model(model: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    rows, train_ids = prepare_rows(spec)
    selection_path = Path(spec["selection_analysis"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    params = selection["result"]["selections"][0]["params"]
    predictions, replay = fit_frozen_policy(rows, train_ids, params)
    template_ids = list(spec["templates"])
    per_template = summarize_templates(rows, predictions, template_ids)
    pai_gamma = float(spec["pai_gamma"])
    pai_per_template = summarize_templates(
        rows, pai_predictions(rows, pai_gamma), template_ids
    )
    return {
        "model": model,
        "selection_analysis": str(selection_path),
        "selection_analysis_sha256": sha256(selection_path),
        "fit_template": "contrastive",
        "test_templates": template_ids,
        "test_template_refit": False,
        "statistics": {
            "mcnemar": "exact two-sided",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "pair",
        },
        "fit_replay": replay,
        "per_template": per_template,
        "delta_ranges": aggregate_range(per_template),
        "frozen_pai": {
            "gamma": pai_gamma,
            "beta": 0.1,
            "selection_source": "native 70-pair development split",
            "test_template_refit": False,
            "per_template": pai_per_template,
            "delta_ranges": aggregate_range(pai_per_template),
        },
        "direction_only_control": direction_only_control(rows, template_ids),
        "input_sha256s": {
            path: sha256(Path(path))
            for path in (
                [spec["train_cs_first"], spec["train_cf_first"]]
                + [
                    path
                    for variants in spec["templates"].values()
                    for path in variants.values()
                ]
            )
        },
    }


def primary_multiplicity(experiments: Mapping[str, Any]) -> Dict[str, Any]:
    raw: Dict[str, float] = {}
    for model, experiment in experiments.items():
        for template, values in experiment["per_template"].items():
            for variant, by_side in values["by_variant"].items():
                for side, metrics in by_side.items():
                    raw[f"{model}/{template}/{variant}/{side}"] = float(
                        metrics["mcnemar_p"]
                    )
    return {
        "family": "two checkpoints x four templates x two answer orders x CF/CS endpoints",
        "raw": raw,
        "holm_adjusted": holm_adjust(raw),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qa_template_stress_v2.json")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    requested = set(args.model or config["models"])
    unknown = requested - set(config["models"])
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    experiments = {
        model: run_model(model, spec)
        for model, spec in config["models"].items()
        if model in requested
    }
    payload = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "protocol": config["evaluation"],
        "experiments": experiments,
        "primary_multiplicity": primary_multiplicity(experiments),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("model\ttemplate\tvariant\tdCF\tdCS\tCF_joint")
    for model, experiment in payload["experiments"].items():
        for template_id, template in experiment["per_template"].items():
            for variant in VARIANTS:
                values = template["by_variant"][variant]
                print(
                    f"{model}\t{template_id}\t{variant}"
                    f"\t{values['counterfactual']['delta']:+.4f}"
                    f"\t{values['commonsense']['delta']:+.4f}"
                    f"\t{template['paired']['counterfactual']['method_joint']:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
