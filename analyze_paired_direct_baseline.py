#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from analyze_candidate_baseline_frozen_transfer import _bootstrap, _sha256, _side_metrics
from analyze_candidate_baseline_sweep import _correct, _read_jsonl
from evaluate_cdh_bench import _candidate_key_from_prediction


def _index(rows: List[Dict[str, Any]], task: str) -> Dict[tuple, Dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("task") != task:
            continue
        key = (str(row.get("pair_id")), str(row.get("side")))
        if key in output:
            raise ValueError(f"duplicate row: {key}")
        output[key] = row
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-block", default="pai")
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--task", choices=("qa", "mc"), required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    baseline = _index(_read_jsonl(args.baseline), args.task)
    method_paths = [str(path) for path in args.method]
    method = _index(
        [record for path in method_paths for record in _read_jsonl(path)], args.task
    )
    if set(baseline) != set(method):
        missing_method = sorted(set(baseline) - set(method))
        missing_baseline = sorted(set(method) - set(baseline))
        raise SystemExit(
            f"key mismatch: missing_method={missing_method[:5]} missing_baseline={missing_baseline[:5]}"
        )

    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key in sorted(baseline):
        pair_id, side = key
        baseline_record = baseline[key]
        method_record = method[key]
        block = baseline_record.get(args.baseline_block)
        if not isinstance(block, dict):
            raise SystemExit(f"missing baseline block {args.baseline_block!r} at {key}")
        baseline_key = block.get("baseline_key") or _candidate_key_from_prediction(
            args.task, block.get("baseline_pred", "")
        )
        gt = str(baseline_record["gt"])
        if str(method_record.get("gt")) != gt:
            raise SystemExit(f"ground-truth mismatch at {key}")
        row = {
            "pair_id": pair_id,
            "side": side,
            "baseline_correct": _correct(args.task, baseline_key, gt),
            "method_correct": bool(method_record.get("correct")),
        }
        by_pair.setdefault(pair_id, {})[side] = row

    incomplete = [pair_id for pair_id, sides in by_pair.items() if set(sides) != {"counterfactual", "commonsense"}]
    if incomplete:
        raise SystemExit(f"incomplete pairs: {incomplete[:5]}")
    cf = _side_metrics([sides["counterfactual"] for sides in by_pair.values()])
    cs = _side_metrics([sides["commonsense"] for sides in by_pair.values()])
    payload = {
        "version": "paired_direct_baseline_v2",
        "method_name": args.method_name,
        "task": args.task,
        "baseline": args.baseline,
        "baseline_sha256": _sha256(args.baseline),
        "baseline_block": args.baseline_block,
        "methods": method_paths,
        "method_sha256s": {path: _sha256(path) for path in method_paths},
        "pairs": len(by_pair),
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
