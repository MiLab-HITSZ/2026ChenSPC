#!/usr/bin/env python3
"""Condition frozen QA correction on sample-level no-image semantic equivariance."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from analyze_cp_vbc_bayes_path_cv import normalize_key
from analyze_hierarchical_eb_lambda_cv import resolved_baseline_key
from analyze_qa_template_stress import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    SIDES,
    VARIANTS,
    fit_frozen_policy,
    prepare_rows,
    prior_top,
    sha256,
)


VERSION = "qa_semantic_equivariance_conditioned_v1"


def correct(prediction: str, row: Mapping[str, Any]) -> bool:
    return normalize_key(prediction) == normalize_key(row.get("gt"))


def template_rows(
    rows: Sequence[Mapping[str, Any]], template_id: str
) -> Dict[str, Dict[str, Dict[str, int]]]:
    output: Dict[str, Dict[str, Dict[str, int]]] = {}
    for index, row in enumerate(rows):
        dataset = str(row.get("_dataset", ""))
        prefix = f"test_{template_id}_"
        if not dataset.startswith(prefix):
            continue
        variant = dataset[len(prefix) :]
        if variant not in VARIANTS or row.get("side") not in SIDES:
            continue
        pair_id = str(row.get("pair_id"))
        output.setdefault(pair_id, {}).setdefault(variant, {})[
            str(row["side"])
        ] = index
    for pair_id, variants in output.items():
        for variant in VARIANTS:
            if set(variants.get(variant, {})) != set(SIDES):
                raise ValueError(
                    f"incomplete rows for {template_id}/{pair_id}/{variant}"
                )
    return output


def classify_prior_pattern(
    rows: Sequence[Mapping[str, Any]], pair_rows: Mapping[str, Mapping[str, int]]
) -> Dict[str, Any]:
    tops: Dict[str, str] = {}
    for variant in VARIANTS:
        values = {
            prior_top(rows[index]) for index in pair_rows[variant].values()
        }
        if len(values) != 1:
            return {
                "pattern": "side_inconsistent",
                "aligned_equivariant": False,
                "complementary": False,
                "tops": {variant: sorted(values)},
            }
        tops[variant] = next(iter(values))
    if tops == {"cs_first": "yes", "cf_first": "no"}:
        pattern = "aligned_equivariant"
    elif tops == {"cs_first": "no", "cf_first": "yes"}:
        pattern = "inverted_equivariant"
    elif tops["cs_first"] == "yes":
        pattern = "fixed_yes"
    else:
        pattern = "fixed_no"
    return {
        "pattern": pattern,
        "aligned_equivariant": pattern == "aligned_equivariant",
        "complementary": tops["cs_first"] != tops["cf_first"],
        "tops": tops,
    }


def pair_effect(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    pair_rows: Mapping[str, Mapping[str, int]],
    side: str,
) -> float:
    effects = []
    for variant in VARIANTS:
        index = pair_rows[variant][side]
        effects.append(
            float(correct(predictions[index], rows[index]))
            - float(correct(resolved_baseline_key(dict(rows[index])), rows[index]))
        )
    return float(np.mean(effects))


def bootstrap_ci(values: Sequence[float], label: str) -> List[float]:
    if not values:
        return [0.0, 0.0]
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{label}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(data), size=(BOOTSTRAP_DRAWS, len(data)))
    draws = data[indices].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def row_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    ids: Iterable[int],
) -> Dict[str, Any]:
    chosen = list(ids)
    baseline_correct = [
        correct(resolved_baseline_key(dict(rows[index])), rows[index])
        for index in chosen
    ]
    method_correct = [correct(predictions[index], rows[index]) for index in chosen]
    repairs = sum(
        not before and after
        for before, after in zip(baseline_correct, method_correct)
    )
    harms = sum(
        before and not after
        for before, after in zip(baseline_correct, method_correct)
    )
    count = len(chosen)
    return {
        "n_rows": count,
        "baseline_acc": sum(baseline_correct) / count if count else None,
        "method_acc": sum(method_correct) / count if count else None,
        "delta": (
            (sum(method_correct) - sum(baseline_correct)) / count
            if count
            else None
        ),
        "repairs": repairs,
        "harms": harms,
    }


def conditioned_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    pair_lookup: Mapping[str, Mapping[str, Mapping[str, int]]],
    pair_ids: Set[str],
    label: str,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {"n_pairs": len(pair_ids), "by_variant": {}, "pooled": {}}
    for variant in VARIANTS:
        output["by_variant"][variant] = {}
        for side in SIDES:
            ids = [pair_lookup[pair_id][variant][side] for pair_id in sorted(pair_ids)]
            output["by_variant"][variant][side] = row_metrics(rows, predictions, ids)
    for side in SIDES:
        ids = [
            pair_lookup[pair_id][variant][side]
            for pair_id in sorted(pair_ids)
            for variant in VARIANTS
        ]
        metrics = row_metrics(rows, predictions, ids)
        effects = [
            pair_effect(rows, predictions, pair_lookup[pair_id], side)
            for pair_id in sorted(pair_ids)
        ]
        metrics["pair_cluster_delta_ci95"] = bootstrap_ci(
            effects, f"{label}:{side}"
        )
        output["pooled"][side] = metrics
    return output


def analyze_template(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    template_id: str,
    model: str,
) -> Dict[str, Any]:
    lookup = template_rows(rows, template_id)
    patterns = {
        pair_id: classify_prior_pattern(rows, pair_rows)
        for pair_id, pair_rows in lookup.items()
    }
    aligned = {
        pair_id
        for pair_id, value in patterns.items()
        if value["aligned_equivariant"]
    }
    not_aligned = set(lookup) - aligned
    by_pattern = {
        pattern: {
            pair_id
            for pair_id, value in patterns.items()
            if value["pattern"] == pattern
        }
        for pattern in sorted({value["pattern"] for value in patterns.values()})
    }
    return {
        "n_pairs": len(lookup),
        "aligned_equivariant_rate": len(aligned) / len(lookup),
        "complementary_rate": sum(
            value["complementary"] for value in patterns.values()
        )
        / len(lookup),
        "pattern_counts": {
            pattern: len(pair_ids) for pattern, pair_ids in by_pattern.items()
        },
        "conditioned": {
            "aligned_equivariant": conditioned_metrics(
                rows,
                predictions,
                lookup,
                aligned,
                f"{model}:{template_id}:aligned",
            ),
            "not_aligned": conditioned_metrics(
                rows,
                predictions,
                lookup,
                not_aligned,
                f"{model}:{template_id}:not_aligned",
            ),
        },
        "by_pattern": {
            pattern: conditioned_metrics(
                rows,
                predictions,
                lookup,
                pair_ids,
                f"{model}:{template_id}:{pattern}",
            )
            for pattern, pair_ids in by_pattern.items()
        },
    }


def run_model(model: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    rows, train_ids = prepare_rows(spec)
    selection_path = Path(spec["selection_analysis"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    params = selection["result"]["selections"][0]["params"]
    predictions, replay = fit_frozen_policy(rows, train_ids, params)
    return {
        "model": model,
        "selection_analysis": str(selection_path),
        "selection_analysis_sha256": sha256(selection_path),
        "test_template_refit": False,
        "conditioning_uses_labels": False,
        "conditioning_definition": (
            "aligned_equivariant iff no-image top is yes for the ordinary-first "
            "claim and no for the atypical-first claim on both image sides"
        ),
        "fit_replay": replay,
        "templates": {
            template_id: analyze_template(rows, predictions, template_id, model)
            for template_id in spec["templates"]
        },
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):+.4f}"

    print("model\ttemplate\tgroup\tn\tdCF\tdCS\tCF_R/H\tCS_R/H")
    for model, experiment in payload["experiments"].items():
        for template, result in experiment["templates"].items():
            for group, metrics in result["conditioned"].items():
                cf = metrics["pooled"]["counterfactual"]
                cs = metrics["pooled"]["commonsense"]
                print(
                    f"{model}\t{template}\t{group}\t{metrics['n_pairs']}"
                    f"\t{fmt(cf['delta'])}\t{fmt(cs['delta'])}"
                    f"\t{cf['repairs']}/{cf['harms']}"
                    f"\t{cs['repairs']}/{cs['harms']}"
                )


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
    payload = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "pair",
        "experiments": {
            model: run_model(model, spec)
            for model, spec in config["models"].items()
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
