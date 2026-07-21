#!/usr/bin/env python3
"""Compare the paired full-precision baseline and official REVIS generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cprc_vlm_runtime import extract_first_letter, extract_yes_no, score_answer


def exact_mcnemar(repairs: int, harms: int) -> float:
    discordant = repairs + harms
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(repairs, harms) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                revis = row.get("revis") or {}
                if row.get("status") != "ok" or "baseline_prediction" not in revis:
                    continue
                key = (str(row.get("pair_id")), str(row.get("task")), str(row.get("side")))
                if key in seen:
                    raise ValueError(f"duplicate official-REVIS row: {key}")
                seen.add(key)
                row = dict(row)
                row["baseline_correct"] = bool(
                    score_answer(row["task"], revis["baseline_prediction"], row["gt"])
                )
                row["steered_correct"] = bool(row["correct"])
                baseline_prediction = revis["baseline_prediction"]
                steered_prediction = revis.get("steered_prediction", row.get("pred"))
                if row["task"] == "mc":
                    baseline_key = extract_first_letter(baseline_prediction)
                    steered_key = extract_first_letter(steered_prediction)
                elif row["task"] == "qa":
                    baseline_key = extract_yes_no(baseline_prediction)
                    steered_key = extract_yes_no(steered_prediction)
                else:
                    baseline_key = baseline_prediction.strip()
                    steered_key = str(steered_prediction or "").strip()
                row["baseline_key"] = baseline_key
                row["steered_key"] = steered_key
                row["changed"] = baseline_key != steered_key
                rows.append(row)
    return rows


def paired_bootstrap_ci(
    rows: list[dict[str, Any]], draws: int, seed: int
) -> list[float] | None:
    if not rows or draws <= 0:
        return None
    differences = np.asarray(
        [int(row["steered_correct"]) - int(row["baseline_correct"]) for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(draws, len(rows)))
    samples = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def summarize(
    rows: Iterable[dict[str, Any]], bootstrap: int = 0, seed: int = 0
) -> dict[str, Any]:
    selected = list(rows)
    baseline_correct = sum(row["baseline_correct"] for row in selected)
    steered_correct = sum(row["steered_correct"] for row in selected)
    repairs = sum((not row["baseline_correct"]) and row["steered_correct"] for row in selected)
    harms = sum(row["baseline_correct"] and (not row["steered_correct"]) for row in selected)
    transitions: dict[str, int] = defaultdict(int)
    observed_positions = 0
    triggered_positions = 0
    risk_sum = 0.0
    rows_with_gate_activity = 0
    for row in selected:
        if row.get("changed"):
            transitions[f"{row.get('baseline_key')}->{row.get('steered_key')}"] += 1
        observation = (row.get("revis") or {}).get("gate_observation") or {}
        positions = int(observation.get("positions") or 0)
        triggered = int(observation.get("triggered_positions") or 0)
        if triggered:
            rows_with_gate_activity += 1
        observed_positions += positions
        triggered_positions += triggered
        risk_sum += float(observation.get("risk_sum") or 0.0)
    return {
        "n": len(selected),
        "baseline_accuracy": baseline_correct / len(selected) if selected else None,
        "official_revis_accuracy": steered_correct / len(selected) if selected else None,
        "delta_accuracy": (steered_correct - baseline_correct) / len(selected) if selected else None,
        "repairs": repairs,
        "harms": harms,
        "changed_predictions": sum(row["changed"] for row in selected),
        "answer_transitions": dict(sorted(transitions.items())),
        "gate": {
            "rows_with_activity": rows_with_gate_activity,
            "rows_with_activity_rate": rows_with_gate_activity / len(selected) if selected else None,
            "positions": observed_positions,
            "triggered_positions": triggered_positions,
            "trigger_rate": triggered_positions / observed_positions if observed_positions else None,
            "risk_mean": risk_sum / observed_positions if observed_positions else None,
        },
        "mcnemar_p": exact_mcnemar(repairs, harms),
        "delta_accuracy_ci95": paired_bootstrap_ci(selected, bootstrap, seed),
    }


def group_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    source_paths = [Path(path) for path in args.results]
    rows = load_rows(source_paths)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    subcategories: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["side"])].append(row)
        subcategories[(row["task"], row["side"], row["subcategory"])].append(row)
    result = {
        "sources": args.results,
        "rows": len(rows),
        "overall": {
            f"{task}.{side}": summarize(
                values,
                max(0, args.bootstrap),
                group_seed(args.seed, f"{task}.{side}"),
            )
            for (task, side), values in sorted(grouped.items())
        },
        "by_subcategory": {
            f"{task}.{side}.{subcategory}": summarize(
                values,
                max(0, args.bootstrap),
                group_seed(args.seed, f"{task}.{side}.{subcategory}"),
            )
            for (task, side, subcategory), values in sorted(subcategories.items())
        },
        "bootstrap": {"draws": max(0, args.bootstrap), "seed": args.seed},
        "changed_cases": [
            {
                "pair_id": row.get("pair_id"),
                "task": row.get("task"),
                "side": row.get("side"),
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "gt": row.get("gt"),
                "baseline_key": row.get("baseline_key"),
                "steered_key": row.get("steered_key"),
                "baseline_correct": row.get("baseline_correct"),
                "steered_correct": row.get("steered_correct"),
                "gate_observation": (row.get("revis") or {}).get("gate_observation") or {},
            }
            for row in rows
            if row.get("changed")
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
