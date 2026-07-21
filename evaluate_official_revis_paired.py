#!/usr/bin/env python3
"""Run the released REVIS intervention on a paired binary-QA benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from cprc_vlm_runtime import (
    generate_official_revis,
    load_model_specs,
    score_answer,
)


TaskKey = Tuple[str, str, str]


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_path(images_root: str, subcategory: str, pair_id: str, side: str) -> str:
    subdirectory = subcategory.replace(" ", "_").replace("/", "_")
    pair_directory = pair_id.replace(" ", "_")
    filename = "counterfactual.png" if side == "counterfactual" else "commonsense.png"
    return str(Path(images_root) / subdirectory / pair_directory / filename)


def completed_keys(path: Path) -> set[TaskKey]:
    if not path.exists():
        return set()
    return {
        (str(row["pair_id"]), str(row["task"]), str(row["side"]))
        for row in read_jsonl(path)
        if row.get("status") == "ok"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--models", default="configs/qwen_32b_instruct_bf16_fair.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vector-file", required=True)
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--risk-gamma", type=float, default=1.0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")
    specs = load_model_specs(args.models)
    if len(specs) != 1:
        raise SystemExit("exactly one transformers model is required")
    spec = specs[0]
    items = [
        item
        for index, item in enumerate(read_jsonl(Path(args.dataset)))
        if index % args.shard_count == args.shard_index
    ]
    if args.limit > 0:
        items = items[: args.limit]
    output = Path(args.output)
    completed = completed_keys(output)
    total = len(items) * 2
    position = 0

    for item in items:
        pair_id = str(item.get("pair_id") or "")
        subcategory = str(item.get("subcategory") or "")
        question = str((item.get("direct_qa") or {}).get("question") or "")
        for side in ("commonsense", "counterfactual"):
            position += 1
            key = (pair_id, "qa", side)
            if key in completed:
                continue
            current_image = image_path(args.images_root, subcategory, pair_id, side)
            ground_truth = str((item.get("direct_qa") or {}).get(f"{side}_gt") or "")
            started = time.time()
            try:
                baseline, baseline_raw = generate_official_revis(
                    spec,
                    question,
                    current_image,
                    args.vector_file,
                    args.calibration_file,
                    args.repo_root,
                    False,
                    args.alpha,
                    args.risk_gamma,
                )
                steered, revis_raw = generate_official_revis(
                    spec,
                    question,
                    current_image,
                    args.vector_file,
                    args.calibration_file,
                    args.repo_root,
                    True,
                    args.alpha,
                    args.risk_gamma,
                )
                status = "ok"
                error = None
            except Exception as exception:
                baseline, steered, baseline_raw, revis_raw = "", "", {}, {}
                status = "error"
                error = str(exception)
            revis_raw = {
                **revis_raw,
                "baseline_prediction": baseline,
                "steered_prediction": steered,
            }
            record = {
                "model_name": spec.name,
                "model": spec.model,
                "pair_id": pair_id,
                "cv_group": item.get("cv_group"),
                "category": item.get("category"),
                "subcategory": subcategory,
                "task": "qa",
                "side": side,
                "image_path": current_image,
                "question": question,
                "gt": ground_truth,
                "status": status,
                "pred": steered,
                "correct": score_answer("qa", steered, ground_truth) if status == "ok" else None,
                "latency_ms": int((time.time() - started) * 1000),
                "revis": revis_raw if status == "ok" else None,
                "raw": {
                    "baseline_raw": baseline_raw,
                    "revis": revis_raw,
                }
                if status == "ok"
                else None,
                "error": error,
            }
            append_jsonl(output, record)
            if position % 50 == 0:
                print(f"shard {args.shard_index}: {position}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
