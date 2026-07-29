#!/usr/bin/env python3
"""Build one collision-free manifest for the external NoLan evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SOURCES = (
    (
        "Visual CounterFact",
        ("result/visual_counterfact_frozen_v1/qwen/results.jsonl",),
    ),
    (
        "HallusionBench",
        (
            "result/hallusionbench_qa_transfer_full_32b_bf16_native_candidate_scores/shard0.jsonl",
            "result/hallusionbench_qa_transfer_full_32b_bf16_native_candidate_scores/shard1.jsonl",
        ),
    ),
    (
        "ConflictVIS",
        (
            "result/cprc_external/conflictvis_full_bf16_shard0.jsonl",
            "result/cprc_external/conflictvis_full_bf16_shard1.jsonl",
        ),
    ),
    (
        "POPE",
        (
            "result/pope_native_qwen32b_bf16/shard0.jsonl",
            "result/pope_native_qwen32b_bf16/shard1.jsonl",
        ),
    ),
    (
        "POPEv2",
        (
            "result/popev2_qa_transfer_32b_instruct/Qwen3-VL-32B-Instruct-transformers-maxtok48/results.jsonl",
        ),
    ),
)


def load_rows(paths: Sequence[str]) -> Iterable[Dict[str, Any]]:
    for value in paths:
        path = Path(value)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def normalized_row(benchmark: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(row)
    source_pair_id = str(value.get("pair_id") or value.get("source_id") or "")
    if not source_pair_id:
        raise ValueError(f"{benchmark} row has no stable identity")
    value["benchmark"] = benchmark
    value["source_pair_id"] = source_pair_id
    value["pair_id"] = f"{benchmark}::{source_pair_id}"
    return value


def build(output: Path) -> Dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    identities: set[tuple[str, str, str]] = set()
    total = 0
    with output.open("w", encoding="utf-8") as handle:
        for benchmark, paths in SOURCES:
            count = 0
            for source_row in load_rows(paths):
                row = normalized_row(benchmark, source_row)
                if str(row.get("status") or "") != "ok":
                    continue
                identity = (
                    str(row["pair_id"]),
                    str(row.get("task") or ""),
                    str(row.get("side") or ""),
                )
                if identity in identities:
                    raise ValueError(f"duplicate row identity: {identity}")
                identities.add(identity)
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                count += 1
                total += 1
            counts[benchmark] = count
    return {"output": str(output), "rows": total, "benchmarks": counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="result/official_nolan_external/qwen32b_external_input.jsonl",
    )
    args = parser.parse_args()
    print(json.dumps(build(Path(args.output)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
