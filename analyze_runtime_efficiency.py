#!/usr/bin/env python3
"""Summarize measured latency and auditable inference-call structure."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    rows.append(row)
    return rows


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> Dict[str, float | int | None]:
    data = [float(value) for value in values]
    return {
        "n": len(data),
        "mean": statistics.fmean(data) if data else None,
        "median": percentile(data, 0.5),
        "p05": percentile(data, 0.05),
        "p95": percentile(data, 0.95),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def call_structure(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cp_rows = [row for row in rows if isinstance(row.get("cp_vbc"), dict)]
    revis_rows = [row for row in rows if isinstance(row.get("revis"), dict)]
    if cp_rows:
        candidate_counts = [len(row["cp_vbc"].get("candidates", [])) for row in cp_rows]
        teacher_forced_calls = [2 * count for count in candidate_counts]
        return {
            "method": "H-EB CPRC score collection",
            "autoregressive_generations_per_row": 1,
            "teacher_forced_candidate_forwards_per_row": distribution(teacher_forced_calls),
            "candidate_count_per_row": distribution(candidate_counts),
            "post_score_h_eb_model_forwards": 0,
            "note": "Each candidate is scored once with the image and once without it; H-EB posterior calibration/reranking is CPU-side after score collection.",
        }
    if revis_rows:
        gate_forward_calls = [
            float(row["revis"].get("gate_observation", {}).get("forward_calls", 0))
            for row in revis_rows
        ]
        return {
            "method": "official REVIS",
            "autoregressive_generations_per_row": 2,
            "teacher_forced_candidate_forwards_per_row": 0,
            "hooked_steered_generation_forward_calls": distribution(gate_forward_calls),
            "note": "One unsteered baseline generation plus one official token-hook steered generation.",
        }
    return {
        "method": "baseline",
        "autoregressive_generations_per_row": 1,
        "teacher_forced_candidate_forwards_per_row": 0,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    latency = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    answer = [
        float(row["answer_latency_ms"])
        for row in rows
        if row.get("answer_latency_ms") is not None
    ]
    overhead = [
        max(0.0, float(row["latency_ms"]) - float(row["answer_latency_ms"]))
        for row in rows
        if row.get("latency_ms") is not None and row.get("answer_latency_ms") is not None
    ]
    serial_seconds = sum(latency) / 1000.0
    return {
        "rows": len(rows),
        "latency_ms": distribution(latency),
        "baseline_generation_latency_ms": distribution(answer),
        "method_overhead_ms": distribution(overhead),
        "serial_wall_seconds": serial_seconds,
        "serial_rows_per_hour": (len(latency) * 3600.0 / serial_seconds) if serial_seconds else None,
        "call_structure": call_structure(rows),
        "exact_generated_tokens_available": False,
        "tokens_per_second": None,
        "peak_gpu_memory_available": False,
        "peak_gpu_memory": None,
    }


def summarize_file(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row.get("task") or "unknown"), []).append(row)
    return {
        "path": str(path),
        "rows": len(rows),
        "all": summarize_rows(rows),
        "by_task": {task: summarize_rows(task_rows) for task, task_rows in sorted(by_task.items())},
    }


def parse_labelled(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_labelled, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {label: summarize_file(path) for label, path in args.input}
    result = {
        "version": "cdh_runtime_efficiency_v1",
        "units": {"latency": "milliseconds", "throughput": "rows/hour"},
        "measurement_boundary": "Per-row evaluator timing; model loading and offline H-EB fitting are excluded.",
        "limitations": [
            "Existing frozen logs do not contain exact generated-token counts.",
            "Existing frozen logs do not contain peak GPU-memory counters.",
            "Rows are serial (parallel=1); throughput is sum-of-row-time throughput, not a serving benchmark.",
        ],
        "runs": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
