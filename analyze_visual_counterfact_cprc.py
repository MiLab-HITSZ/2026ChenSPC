#!/usr/bin/env python3
"""Frozen CDH-to-Visual-CounterFact transfer with matched controls."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_cprc_gate_pareto import paired_counts
from analyze_cprc_shift_matched_controls import CORE_FAMILIES, run_core_families
from analyze_hierarchical_eb_lambda_cv import (
    dataset_metrics,
    exact_mcnemar_pvalue,
    resolved_baseline_key,
)


VERSION = "visual_counterfact_bprc_transfer_v1"
SIDES = ("counterfactual", "commonsense")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(spec: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[int]]:
    rows: List[Dict[str, Any]] = []
    for label, task, path in (
        ("dev_mc", "mc", spec["dev_mc"]),
        ("dev_qa", "qa", spec["dev_qa"]),
    ):
        for source in read_jsonl(Path(path), task):
            row = copy.deepcopy(source)
            row["_dataset"] = label
            row["_task"] = task
            rows.append(row)
    target_ids = []
    for source in read_jsonl(Path(spec["target"]), "mc"):
        row = copy.deepcopy(source)
        row["_dataset"] = "test_vcf_mc"
        row["_task"] = "mc"
        target_ids.append(len(rows))
        rows.append(row)
    return rows, target_ids


def bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    ids: Sequence[int],
    side: str,
    samples: int,
    seed: int,
) -> List[float]:
    values = [
        int(normalize_key(predictions[index]) == normalize_key(rows[index].get("gt")))
        - int(
            resolved_baseline_key(dict(rows[index]))
            == normalize_key(rows[index].get("gt"))
        )
        for index in ids
        if rows[index].get("side") == side
    ]
    if not values:
        return [0.0, 0.0]
    rng = random.Random(int(seed))
    draws = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(max(1, int(samples)))
    )
    return [
        float(draws[int(0.025 * (len(draws) - 1))]),
        float(draws[int(0.975 * (len(draws) - 1))]),
    ]


def holm_adjust(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (float(item[1]), item[0]))
    count = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in values}


def mechanism_diagnostic(
    rows: Sequence[Mapping[str, Any]], ids: Sequence[int]
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for side in SIDES:
        selected = [index for index in ids if rows[index].get("side") == side]
        prior_typical = 0
        baseline_errors = 0
        error_matches_prior = 0
        error_matches_typical = 0
        for index in selected:
            row = rows[index]
            prior = normalize_key((row.get("cp_vbc") or {}).get("prior_top"))
            baseline = resolved_baseline_key(dict(row))
            gt = normalize_key(row.get("gt"))
            typical = normalize_key(row.get("typical_key"))
            prior_typical += prior == typical
            if baseline != gt:
                baseline_errors += 1
                error_matches_prior += baseline == prior
                error_matches_typical += baseline == typical
        output[side] = {
            "n": len(selected),
            "prior_top_is_typical": prior_typical,
            "prior_top_is_typical_rate": prior_typical / len(selected) if selected else 0.0,
            "baseline_errors": baseline_errors,
            "error_matches_prior_top": error_matches_prior,
            "error_matches_prior_top_rate": (
                error_matches_prior / baseline_errors if baseline_errors else 0.0
            ),
            "error_matches_typical_answer": error_matches_typical,
            "error_matches_typical_answer_rate": (
                error_matches_typical / baseline_errors if baseline_errors else 0.0
            ),
        }
    return output


def selected_result(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    ids: Sequence[int],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    metrics = dataset_metrics(rows, predictions, ids)["test_vcf_mc"]
    counts = paired_counts(rows, predictions, ids)["test_vcf_mc"]
    for side_index, side in enumerate(SIDES):
        repairs = int(counts[side]["repairs"])
        harms = int(counts[side]["harms"])
        counts[side]["exact_mcnemar_p_two_sided"] = exact_mcnemar_pvalue(
            repairs, harms
        )
        counts[side]["delta_ci95"] = bootstrap_delta(
            rows,
            predictions,
            ids,
            side,
            bootstrap,
            seed + side_index,
        )
    return {"metrics": metrics, "paired": counts}


def run_model(
    model: str, spec: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    rows, target_ids = load_rows(spec)
    expected = int(config["expected_target_rows"])
    if len(target_ids) != expected:
        raise ValueError(f"{model}: expected {expected} target rows, found {len(target_ids)}")
    selector, geometry = run_core_families(rows, config)
    output: Dict[str, Any] = {
        "model": model,
        "target_rows": len(target_ids),
        "target_pairs": len({rows[index]["pair_id"] for index in target_ids}),
        "mechanism": mechanism_diagnostic(rows, target_ids),
        "selected": {},
    }
    strata = {
        "overall": target_ids,
        "color": [
            index for index in target_ids if rows[index].get("source_split") == "color"
        ],
        "size": [
            index for index in target_ids if rows[index].get("source_split") == "size"
        ],
    }
    for family in CORE_FAMILIES:
        output["selected"][family] = {}
        for budget, candidate in sorted(
            selector.best.get(family, {}).items(), key=lambda item: float(item[0])
        ):
            predictions = candidate["predictions"]
            output["selected"][family][budget] = {
                "params": candidate["params"],
                "development_metrics": candidate["development_metrics"],
                "by_stratum": {
                    name: selected_result(
                        rows,
                        predictions,
                        ids,
                        int(config["bootstrap"]),
                        int(config["seed"]) + stratum_index * 10,
                    )
                    for stratum_index, (name, ids) in enumerate(strata.items())
                },
            }
    return output


def primary_holm(
    experiments: Mapping[str, Any], primary_budget: str
) -> Dict[str, Any]:
    raw = {}
    for model, experiment in experiments.items():
        paired = experiment["selected"]["bprc_full"][primary_budget]["by_stratum"][
            "overall"
        ]["paired"]
        for side in SIDES:
            raw[f"{model}/{side}"] = float(
                paired[side]["exact_mcnemar_p_two_sided"]
            )
    return {
        "family": "two checkpoints x CF/CS overall endpoints",
        "raw": raw,
        "holm_adjusted": holm_adjust(raw),
    }


def print_summary(payload: Mapping[str, Any]) -> None:
    print("model\tfamily\tbudget\tstratum\tdCF\tdCS\tCF_repairs\tCF_harms")
    for model, experiment in payload["experiments"].items():
        for family, by_budget in experiment["selected"].items():
            for budget, selected in by_budget.items():
                for stratum, result in selected["by_stratum"].items():
                    metrics = result["metrics"]
                    paired = result["paired"]["counterfactual"]
                    print(
                        f"{model}\t{family}\t{budget}\t{stratum}"
                        f"\t{metrics['counterfactual']['delta']:+.4f}"
                        f"\t{metrics['commonsense']['delta']:+.4f}"
                        f"\t{paired['repairs']}\t{paired['harms']}"
                    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/visual_counterfact_analysis_v1.json"
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
        "frozen_protocol": config["frozen_protocol"],
        "target_refit": False,
        "experiments": {},
    }
    for model, spec in config["models"].items():
        if model in requested:
            payload["experiments"][model] = run_model(model, spec, config)
    primary_budget = f"{float(config['primary_budget']):.2f}"
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
