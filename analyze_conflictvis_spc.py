#!/usr/bin/env python3
"""Apply frozen CDH EB-SPC and PAI to ConflictVIS candidate-complete tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from analyze_candidate_baseline_sweep import _method_block, _replay_pai
from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_hierarchical_eb_lambda_cv import resolved_baseline_key, run_cv
from analyze_pope_native_spc import exact_mcnemar, multinomial_bootstrap_ci


def _load_target(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        for source in read_jsonl(path, None):
            key = str(source.get("source_id"))
            if key in seen:
                raise ValueError(f"duplicate ConflictVIS row: {key}")
            seen.add(key)
            row = dict(source)
            row["_dataset"] = f"conflictvis_{row['task']}"
            row["_task"] = str(row["task"])
            rows.append(row)
    return rows


def _corrected_predictions(
    rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any]
) -> Dict[Tuple[str, str, str], str]:
    predictions = {
        (str(row["_dataset"]), str(row["pair_id"]), str(row["side"])): resolved_baseline_key(dict(row))
        for row in rows
    }
    for case in result["cases"]:
        dataset = str(case["dataset"])
        if dataset.startswith("conflictvis_"):
            predictions[(dataset, str(case["pair_id"]), str(case["side"]))] = normalize_key(case["prediction"])
    return predictions


def _metrics(labels: Sequence[str], baseline: Sequence[str], method: Sequence[str], draws: int, seed: int) -> Dict[str, Any]:
    repairs = sum(base != label and fixed == label for label, base, fixed in zip(labels, baseline, method))
    harms = sum(base == label and fixed != label for label, base, fixed in zip(labels, baseline, method))
    n = len(labels)
    baseline_correct = sum(base == label for base, label in zip(baseline, labels))
    method_correct = sum(fixed == label for fixed, label in zip(method, labels))
    interventions = sum(base != fixed for base, fixed in zip(baseline, method))
    return {
        "n": n,
        "baseline_accuracy": baseline_correct / n,
        "method_accuracy": method_correct / n,
        "delta_accuracy": (method_correct - baseline_correct) / n,
        "interventions": interventions,
        "intervention_rate": interventions / n,
        "repairs": repairs,
        "harms": harms,
        "mcnemar_exact_p": exact_mcnemar(repairs, harms),
        "delta_accuracy_ci95": multinomial_bootstrap_ci(repairs, harms, n, draws, seed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--dev-mc", required=True)
    parser.add_argument("--dev-qa", required=True)
    parser.add_argument("--pai-mc-selection", required=True)
    parser.add_argument("--pai-qa-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for label, task, path in (("dev_mc", "mc", args.dev_mc), ("dev_qa", "qa", args.dev_qa)):
        for source in read_jsonl(Path(path), task):
            row = dict(source)
            row["_dataset"] = label
            row["_task"] = task
            rows.append(row)
    target = _load_target(Path(path) for path in args.target)
    rows.extend(target)
    result = run_cv(rows, "frozen_transfer", 5, 0.20, True, 64, 1, int(args.seed), "net_utility", 0.5, ("dev_mc", "dev_qa"))
    eb = _corrected_predictions(target, result)
    pai_params = {
        "mc": json.loads(Path(args.pai_mc_selection).read_text(encoding="utf-8"))["selected"]["params"],
        "qa": json.loads(Path(args.pai_qa_selection).read_text(encoding="utf-8"))["selected"]["params"],
    }
    predictions: Dict[str, Dict[str, str]] = {}
    for row in target:
        baseline = resolved_baseline_key(row)
        block = _method_block(row, "pai")
        if block is None:
            raise ValueError(f"missing PAI-compatible candidate scores: {row.get('source_id')}")
        pai = _replay_pai(block, pai_params[str(row["task"])])
        if pai is None:
            raise ValueError(f"PAI replay failed: {row.get('source_id')}")
        key = str(row["source_id"])
        predictions[key] = {
            "baseline": normalize_key(baseline),
            "pai": normalize_key(pai),
            "eb_cprc": normalize_key(eb[(str(row["_dataset"]), str(row["pair_id"]), str(row["side"]))]),
        }

    slices: Dict[str, List[Dict[str, Any]]] = {"overall": list(target)}
    for task in ("qa", "mc"):
        slices[f"task:{task}"] = [row for row in target if row["task"] == task]
    for conflict_target in sorted({str(row["conflict_target"]) for row in target}):
        slices[f"target:{conflict_target}"] = [row for row in target if row["conflict_target"] == conflict_target]
    metrics = {}
    for index, (name, selected) in enumerate(slices.items()):
        labels = [normalize_key(row["gt"]) for row in selected]
        baseline = [predictions[str(row["source_id"])]["baseline"] for row in selected]
        metrics[name] = {
            method: _metrics(
                labels,
                baseline,
                [predictions[str(row["source_id"])][method] for row in selected],
                int(args.bootstrap),
                int(args.seed) + index * 10 + offset,
            )
            for offset, method in enumerate(("pai", "eb_cprc"), start=1)
        }
    artifact = {
        "version": "conflictvis_frozen_transfer_v1",
        "protocol": "All official candidate-complete YN and MC rows; one image per query; no ConflictVIS label used for fit, gate selection, or PAI parameter selection.",
        "target_sources": args.target,
        "target_rows": len(target),
        "calibration": {"eb_cprc": "CDH 70-pair BF16 development fit", "pai": {"source": "CDH development-only sweeps", "params": pai_params}},
        "metrics": metrics,
        "eb_fit_selection": result["selections"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
