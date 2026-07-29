#!/usr/bin/env python3
"""Select and merge full-decoding fallbacks for invalid first-token answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from evaluate_official_nolan_external import row_identity, summarize


Identity = Tuple[str, str, str]


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def select_fallback_rows(
    source_rows: Sequence[Mapping[str, Any]],
    one_token_rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    invalid = {
        row_identity(row)
        for row in one_token_rows
        if row.get("native_key") is None or row.get("pred_key") is None
    }
    source = {row_identity(row): dict(row) for row in source_rows}
    missing = invalid.difference(source)
    if missing:
        raise ValueError(f"fallback identities missing from source: {sorted(missing)[:3]}")
    return [source[identity] for identity in sorted(invalid)]


def merged_rows(
    one_token_rows: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]],
    nolan_rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    native = {row_identity(row): row for row in native_rows}
    nolan = {row_identity(row): row for row in nolan_rows}
    merged: list[Dict[str, Any]] = []
    for source in one_token_rows:
        row = dict(source)
        identity = row_identity(row)
        needs_fallback = row.get("native_key") is None or row.get("pred_key") is None
        if needs_fallback:
            native_row = native.get(identity)
            nolan_row = nolan.get(identity)
            if native_row is None or nolan_row is None:
                raise ValueError(f"missing full-decoding fallback for {identity}")
            row["one_token"] = {
                "native_pred": row.get("native_pred"),
                "native_key": row.get("native_key"),
                "pred": row.get("pred"),
                "pred_key": row.get("pred_key"),
                "nolan": row.get("nolan"),
            }
            row["native_pred"] = native_row.get("pred")
            row["native_key"] = native_row.get("pred_key")
            row["native_correct"] = bool(native_row.get("correct"))
            row["pred"] = nolan_row.get("pred")
            row["pred_key"] = nolan_row.get("pred_key")
            row["correct"] = bool(nolan_row.get("correct"))
            row["nolan"] = nolan_row.get("nolan")
            row["decoding_fallback"] = "full_generation"
        merged.append(row)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--input", required=True)
    select.add_argument("--one-token-output", required=True)
    select.add_argument("--output", required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--one-token-output", required=True)
    merge.add_argument("--native-output", required=True)
    merge.add_argument("--nolan-output", required=True)
    merge.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "select":
        rows = select_fallback_rows(
            load_jsonl(Path(args.input)),
            load_jsonl(Path(args.one_token_output)),
        )
        write_jsonl(Path(args.output), rows)
        print(json.dumps({"fallback_rows": len(rows), "output": args.output}, indent=2))
        return 0

    rows = merged_rows(
        load_jsonl(Path(args.one_token_output)),
        load_jsonl(Path(args.native_output)),
        load_jsonl(Path(args.nolan_output)),
    )
    output = Path(args.output)
    write_jsonl(output, rows)
    summary = {
        "output": str(output),
        "answer_protocol": "answer_token_with_full_generation_fallback",
        "metrics": summarize(rows),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
