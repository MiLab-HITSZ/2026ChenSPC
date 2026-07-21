#!/usr/bin/env python3
"""Evaluate whether candidate-prior correction follows MC semantics or positions."""

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_spc_shift_matched_controls import (
    CORE_FAMILIES,
    compare_outcomes,
    run_core_families,
)
from analyze_hierarchical_eb_lambda_cv import dataset_metrics, resolved_baseline_key


VERSION = "bprc_mc_option_permutation_v1"
LETTERS = "ABCD"
SIDES = ("counterfactual", "commonsense")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(spec: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    rows: List[Dict[str, Any]] = []
    variant_ids: Dict[str, List[int]] = {}
    for label, task, path in (
        ("dev_mc", "mc", spec["dev_mc"]),
        ("dev_qa", "qa", spec["dev_qa"]),
    ):
        for source in read_jsonl(Path(path), task):
            row = copy.deepcopy(source)
            row["_dataset"] = label
            row["_task"] = task
            row["_variant"] = "development"
            rows.append(row)

    sources = {"original": spec["original_test"], **spec["permuted_tests"]}
    for variant, path in sources.items():
        selected = []
        for source in read_jsonl(Path(path), "mc"):
            row = copy.deepcopy(source)
            row["_dataset"] = f"test_{variant}_mc"
            row["_task"] = "mc"
            row["_variant"] = variant
            selected.append(len(rows))
            rows.append(row)
        variant_ids[variant] = selected
    return rows, variant_ids


def semantic_key(letter: str, new_to_old: Sequence[int]) -> str:
    normalized = normalize_key(letter).upper()
    if normalized not in LETTERS:
        return ""
    return LETTERS[int(new_to_old[LETTERS.index(normalized)])]


def correctness_map(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    ids: Sequence[int],
    side: str,
) -> Dict[Tuple[str, str, str], bool]:
    return {
        (
            str(rows[index].get("_task")),
            str(rows[index].get("pair_id")),
            str(rows[index].get("side")),
        ): normalize_key(predictions[index]) == normalize_key(rows[index].get("gt"))
        for index in ids
        if rows[index].get("side") == side
    }


def paired_variant_test(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    reference_ids: Sequence[int],
    candidate_ids: Sequence[int],
    side: str,
) -> Dict[str, Any]:
    reference = correctness_map(rows, predictions, reference_ids, side)
    candidate = correctness_map(rows, predictions, candidate_ids, side)
    # Variant names differ but pair/task/side identities are deliberately shared.
    return compare_outcomes(reference, candidate)


def paired_method_test(
    rows: Sequence[Mapping[str, Any]],
    baseline: Sequence[str],
    predictions: Sequence[str],
    ids: Sequence[int],
    side: str,
) -> Dict[str, Any]:
    reference = correctness_map(rows, baseline, ids, side)
    candidate = correctness_map(rows, predictions, ids, side)
    return compare_outcomes(reference, candidate)


def holm_adjust(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (float(item[1]), item[0]))
    count = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in values}


def semantic_consistency(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    variant_ids: Mapping[str, Sequence[int]],
    permutations: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    by_identity: Dict[Tuple[str, str], Dict[str, str]] = {}
    gt_by_identity: Dict[Tuple[str, str], Dict[str, str]] = {}
    for variant, ids in variant_ids.items():
        permutation = permutations[variant]
        for index in ids:
            row = rows[index]
            identity = (str(row.get("pair_id")), str(row.get("side")))
            by_identity.setdefault(identity, {})[variant] = semantic_key(
                predictions[index], permutation
            )
            gt_by_identity.setdefault(identity, {})[variant] = semantic_key(
                str(row.get("gt")), permutation
            )

    complete = [
        identity
        for identity, values in by_identity.items()
        if set(values) == set(variant_ids)
    ]
    inconsistent_gt = [
        identity
        for identity in complete
        if len(set(gt_by_identity[identity].values())) != 1
    ]
    if inconsistent_gt:
        raise AssertionError(f"semantic GT changed under permutation: {inconsistent_gt[:3]}")
    output = {}
    for side in SIDES:
        selected = [identity for identity in complete if identity[1] == side]
        consistent = sum(
            len(set(by_identity[identity].values())) == 1 for identity in selected
        )
        output[side] = {
            "n": len(selected),
            "all_orders_same_semantic_prediction": consistent,
            "semantic_consistency": consistent / len(selected) if selected else 0.0,
        }
    return output


def variant_metrics(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    variant_ids: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    output = {}
    for variant, ids in variant_ids.items():
        label = f"test_{variant}_mc"
        output[variant] = dataset_metrics(rows, predictions, ids)[label]
    return output


def robustness_summary(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    permutations = [name for name in metrics if name != "original"]
    return {
        side: {
            "original_delta": float(metrics["original"][side]["delta"]),
            "permuted_mean_delta": statistics.mean(
                float(metrics[name][side]["delta"]) for name in permutations
            ),
            "permuted_worst_delta": min(
                float(metrics[name][side]["delta"]) for name in permutations
            ),
            "method_accuracy_range": max(
                float(metrics[name][side]["acc"]) for name in metrics
            )
            - min(float(metrics[name][side]["acc"]) for name in metrics),
        }
        for side in SIDES
    }


def run_model(
    model: str,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    permutations: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    rows, variant_ids = load_rows(spec)
    if set(variant_ids) != set(permutations):
        raise ValueError("result variants and permutation manifest do not match")
    selector, geometry = run_core_families(rows, config)
    baseline = [resolved_baseline_key(row) for row in rows]
    output: Dict[str, Any] = {
        "model": model,
        "development_rows": len(geometry.train_ids),
        "test_rows_per_variant": {
            name: len(ids) for name, ids in variant_ids.items()
        },
        "baseline": {
            "metrics": variant_metrics(rows, baseline, variant_ids),
            "semantic_consistency": semantic_consistency(
                rows, baseline, variant_ids, permutations
            ),
        },
        "selected": {},
    }
    for family in CORE_FAMILIES:
        output["selected"][family] = {}
        for budget, candidate in sorted(
            selector.best.get(family, {}).items(), key=lambda item: float(item[0])
        ):
            predictions = candidate["predictions"]
            metrics = variant_metrics(rows, predictions, variant_ids)
            output["selected"][family][budget] = {
                "params": candidate["params"],
                "development_metrics": candidate["development_metrics"],
                "metrics": metrics,
                "robustness": robustness_summary(metrics),
                "semantic_consistency": semantic_consistency(
                    rows, predictions, variant_ids, permutations
                ),
                "paired_method_vs_baseline": {
                    variant: {
                        side: paired_method_test(
                            rows, baseline, predictions, ids, side
                        )
                        for side in SIDES
                    }
                    for variant, ids in variant_ids.items()
                },
                "paired_permutation_vs_original": {
                    variant: {
                        side: paired_variant_test(
                            rows,
                            predictions,
                            variant_ids["original"],
                            ids,
                            side,
                        )
                        for side in SIDES
                    }
                    for variant, ids in variant_ids.items()
                    if variant != "original"
                },
            }
    return output


def primary_holm(
    experiments: Mapping[str, Any], primary_budget: str
) -> Dict[str, Any]:
    raw: Dict[str, float] = {}
    for model, experiment in experiments.items():
        paired = experiment["selected"]["bprc_full"][primary_budget][
            "paired_method_vs_baseline"
        ]
        for variant, by_side in paired.items():
            for side, result in by_side.items():
                raw[f"{model}/{variant}/{side}"] = float(
                    result["exact_mcnemar_p_two_sided"]
                )
    return {
        "family": "two checkpoints x four MC orders x CF/CS endpoints",
        "budget": primary_budget,
        "raw": raw,
        "holm_adjusted": holm_adjust(raw),
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    print("model\tfamily\tbudget\torder\tdCF\tdCS\tsemantic_consistency_CF")
    for model, result in payload["experiments"].items():
        for family, by_budget in result["selected"].items():
            for budget, selected in by_budget.items():
                consistency = selected["semantic_consistency"]["counterfactual"][
                    "semantic_consistency"
                ]
                for variant, metrics in selected["metrics"].items():
                    print(
                        f"{model}\t{family}\t{budget}\t{variant}"
                        f"\t{metrics['counterfactual']['delta']:+.4f}"
                        f"\t{metrics['commonsense']['delta']:+.4f}"
                        f"\t{consistency:.4f}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/spc_mc_option_permutation_v1.json"
    )
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["permutation_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    permutations = {"original": [0, 1, 2, 3]}
    permutations.update(
        {
            name: artifact["permutation"]
            for name, artifact in manifest["artifacts"].items()
        }
    )
    requested = set(args.model or config["models"])
    unknown = requested - set(config["models"])
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    payload = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "permutation_manifest": str(manifest_path),
        "permutation_manifest_sha256": sha256(manifest_path),
        "semantic_contract": manifest["semantic_contract"],
        "selection_split": "original 70-pair development scores only",
        "experiments": {},
    }
    for model, spec in config["models"].items():
        if model in requested:
            payload["experiments"][model] = run_model(
                model, spec, config, permutations
            )
    primary_budget = f"{max(float(value) for value in config['cs_budgets']):.2f}"
    payload["primary_multiplicity"] = primary_holm(
        payload["experiments"], primary_budget
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
