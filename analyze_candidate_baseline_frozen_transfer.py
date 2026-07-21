#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analyze_candidate_baseline_sweep import (
    REPLAY,
    _candidate_key_from_prediction,
    _correct,
    _method_block,
    _read_jsonl,
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_mcnemar(repairs: int, harms: int) -> float:
    discordant = int(repairs) + int(harms)
    if discordant == 0:
        return 1.0
    smaller = min(int(repairs), int(harms))
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _quantile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _side_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    baseline = sum(int(row["baseline_correct"]) for row in rows)
    method = sum(int(row["method_correct"]) for row in rows)
    repairs = sum(int(not row["baseline_correct"] and row["method_correct"]) for row in rows)
    harms = sum(int(row["baseline_correct"] and not row["method_correct"]) for row in rows)
    return {
        "n": n,
        "baseline_acc": baseline / n,
        "method_acc": method / n,
        "delta": (method - baseline) / n,
        "repairs": repairs,
        "harms": harms,
        "mcnemar_exact_p": _exact_mcnemar(repairs, harms),
    }


def _bootstrap(
    by_pair: Dict[str, Dict[str, Dict[str, Any]]],
    samples: int,
    seed: int,
) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    pair_ids = sorted(by_pair)
    cf_deltas: List[float] = []
    cs_deltas: List[float] = []
    for _ in range(samples):
        selected = [pair_ids[rng.randrange(len(pair_ids))] for _ in pair_ids]
        cf = [by_pair[pair_id]["counterfactual"] for pair_id in selected]
        cs = [by_pair[pair_id]["commonsense"] for pair_id in selected]
        cf_deltas.append(
            sum(int(row["method_correct"]) - int(row["baseline_correct"]) for row in cf) / len(cf)
        )
        cs_deltas.append(
            sum(int(row["method_correct"]) - int(row["baseline_correct"]) for row in cs) / len(cs)
        )
    return {
        "delta_cf_ci95": [_quantile(cf_deltas, 0.025), _quantile(cf_deltas, 0.975)],
        "delta_cs_ci95": [_quantile(cs_deltas, 0.025), _quantile(cs_deltas, 0.975)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-selection", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    selection = json.loads(Path(args.development_selection).read_text(encoding="utf-8"))
    if selection.get("selection_split") != "development_only":
        raise SystemExit("selection artifact is not marked development_only")
    method = str(selection["method"])
    task = str(selection["task"])
    params = dict(selection["selected"]["params"])
    development_inputs = selection.get("inputs") or [selection.get("input")]
    development_inputs = [str(path) for path in development_inputs if path]
    if not development_inputs:
        raise SystemExit("selection artifact has no development inputs")
    development_rows = [
        record
        for path in development_inputs
        for record in _read_jsonl(path)
    ]
    test_paths = [str(path) for path in args.test]
    test_rows = [record for path in test_paths for record in _read_jsonl(path)]
    seen_test_rows = set()
    for record in test_rows:
        if record.get("status") != "ok" or record.get("task") != task:
            continue
        key = (str(record.get("pair_id")), str(record.get("side")))
        if key in seen_test_rows:
            raise SystemExit(f"duplicate test row across shards: {key}")
        seen_test_rows.add(key)
    development_pairs = {str(row.get("pair_id")) for row in development_rows if row.get("task") == task}
    test_pairs = {str(row.get("pair_id")) for row in test_rows if row.get("task") == task}
    overlap = sorted(development_pairs & test_pairs)
    if overlap:
        raise SystemExit(f"development/test pair overlap: {overlap[:5]}")

    evaluated: List[Dict[str, Any]] = []
    for record in test_rows:
        if record.get("status") != "ok" or record.get("task") != task:
            continue
        block = _method_block(record, method)
        if block is None:
            continue
        final_key = REPLAY[method](block, params)
        if final_key is None:
            raise SystemExit("frozen parameters do not match the collected test view configuration")
        baseline_key = block.get("baseline_key") or _candidate_key_from_prediction(
            task, block.get("baseline_pred", "")
        )
        evaluated.append(
            {
                "pair_id": str(record["pair_id"]),
                "side": str(record["side"]),
                "baseline_key": baseline_key,
                "method_key": final_key,
                "baseline_correct": _correct(task, baseline_key, str(record["gt"])),
                "method_correct": _correct(task, final_key, str(record["gt"])),
            }
        )

    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in evaluated:
        by_pair.setdefault(row["pair_id"], {})[row["side"]] = row
    incomplete = [pair_id for pair_id, sides in by_pair.items() if set(sides) != {"counterfactual", "commonsense"}]
    if incomplete:
        raise SystemExit(f"incomplete test pairs: {incomplete[:5]}")
    cf_rows = [sides["counterfactual"] for sides in by_pair.values()]
    cs_rows = [sides["commonsense"] for sides in by_pair.values()]
    cf = _side_metrics(cf_rows)
    cs = _side_metrics(cs_rows)
    payload = {
        "version": "candidate_baseline_frozen_transfer_v2",
        "method": method,
        "task": task,
        "development_selection": args.development_selection,
        "development_selection_sha256": _sha256(args.development_selection),
        "tests": test_paths,
        "test_sha256s": {path: _sha256(path) for path in test_paths},
        "development_pairs": len(development_pairs),
        "test_pairs": len(by_pair),
        "pair_overlap": 0,
        "frozen_params": params,
        "no_test_refit": True,
        "counterfactual": cf,
        "commonsense": cs,
        "utility": {
            "w_0_25": cf["delta"] + 0.25 * cs["delta"],
            "w_0_5": cf["delta"] + 0.5 * cs["delta"],
            "w_1": cf["delta"] + cs["delta"],
            "w_2": cf["delta"] + 2.0 * cs["delta"],
        },
        "bootstrap": {
            "samples": args.bootstrap,
            "seed": args.seed,
            **_bootstrap(by_pair, args.bootstrap, args.seed),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
