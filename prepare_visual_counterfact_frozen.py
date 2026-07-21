#!/usr/bin/env python3
"""Convert the official Visual CounterFact pairs into a frozen two-option MC track."""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pyarrow.parquet as pq
from PIL import Image


VERSION = "visual_counterfact_frozen_v1"
LETTERS = "AB"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer_text(value: Any) -> str:
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = text
    if isinstance(parsed, (list, tuple)):
        values = [str(item).strip() for item in parsed if str(item).strip()]
        if not values:
            raise ValueError(f"empty answer list: {value!r}")
        return " or ".join(values)
    if not str(parsed).strip():
        raise ValueError("empty answer")
    return str(parsed).strip()


def image_bytes(value: Mapping[str, Any]) -> bytes:
    payload = value.get("bytes")
    if payload:
        return bytes(payload)
    path = value.get("path")
    if path:
        return Path(str(path)).read_bytes()
    raise ValueError("parquet image has neither bytes nor path")


def save_png(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes(value))) as image:
        image.convert("RGB").save(path, format="PNG", compress_level=1)


def option_layout(
    split: str, index: int, typical: str, counterfactual: str
) -> Dict[str, Any]:
    # Alternation is fixed before inference and balances the prior answer across A/B.
    typical_first = (index + int(split == "size")) % 2 == 0
    values = [typical, counterfactual] if typical_first else [counterfactual, typical]
    typical_key = "A" if typical_first else "B"
    counterfactual_key = "B" if typical_first else "A"
    return {
        "options": [f"{letter}. {value}" for letter, value in zip(LETTERS, values)],
        "typical_key": typical_key,
        "counterfactual_key": counterfactual_key,
    }


def question(split: str, object_name: str, options: Sequence[str]) -> str:
    if split == "color":
        stem = f"What color is the {object_name} in this image?"
    elif split == "size":
        stem = "Which object is larger in this image?"
    else:
        raise ValueError(f"unknown split: {split}")
    return f"{stem}\n" + "\n".join(options) + "\nAnswer with A or B."


def convert_split(
    split: str, parquet_path: Path, output_root: Path
) -> List[Dict[str, Any]]:
    rows = pq.read_table(parquet_path).to_pylist()
    output: List[Dict[str, Any]] = []
    subcategory = f"VisualCounterFact {split.title()}"
    category = "Attribute Anomalies" if split == "color" else "Relational Anomalies"
    for index, source in enumerate(rows):
        pair_id = f"VCF {split} {index:04d}"
        typical = answer_text(source["correct_answer"])
        counterfactual = answer_text(source["incorrect_answer"])
        layout = option_layout(split, index, typical, counterfactual)
        pair_dir = output_root / "images" / subcategory.replace(" ", "_") / pair_id.replace(" ", "_")
        cs_path = pair_dir / "commonsense.png"
        cf_path = pair_dir / "counterfactual.png"
        save_png(source["original_image"], cs_path)
        save_png(source["counterfact_image"], cf_path)
        prompt = question(split, str(source["object"]), layout["options"])
        for side, image_path, gt in (
            ("commonsense", cs_path, layout["typical_key"]),
            ("counterfactual", cf_path, layout["counterfactual_key"]),
        ):
            output.append(
                {
                    "source_id": f"{pair_id}/{side}",
                    "pair_id": pair_id,
                    "task": "mc",
                    "side": side,
                    "category": category,
                    "subcategory": subcategory,
                    "source_split": split,
                    "source_index": index,
                    "object": str(source["object"]),
                    "image_path": str(image_path),
                    "question": prompt,
                    "candidate_keys": list(LETTERS),
                    "options": layout["options"],
                    "gt": gt,
                    "typical_key": layout["typical_key"],
                    "counterfactual_key": layout["counterfactual_key"],
                    "typical_answer": typical,
                    "counterfactual_answer": counterfactual,
                    "protocol_version": VERSION,
                }
            )
    return output


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    identities = {
        (str(row["pair_id"]), str(row["side"])) for row in rows
    }
    if len(identities) != len(rows):
        raise AssertionError("duplicate pair/side rows")
    pairs = {str(row["pair_id"]) for row in rows}
    if len(rows) != 2 * len(pairs):
        raise AssertionError("every pair must contain two sides")
    output: Dict[str, Any] = {"pairs": len(pairs), "rows": len(rows), "by_split": {}}
    for split in ("color", "size"):
        selected = [row for row in rows if row["source_split"] == split]
        output["by_split"][split] = {
            "pairs": len({row["pair_id"] for row in selected}),
            "rows": len(selected),
            "typical_answer_position": {
                letter: len(
                    {
                        row["pair_id"]
                        for row in selected
                        if row["typical_key"] == letter
                    }
                )
                for letter in LETTERS
            },
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="downloads/visual_counterfact/data")
    parser.add_argument("--output-dir", default="data/visual_counterfact_frozen_v1")
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    source_paths = {
        "color": source_dir / "color-00000-of-00001.parquet",
        "size": source_dir / "size-00000-of-00001.parquet",
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    for split, path in source_paths.items():
        rows.extend(convert_split(split, path, output_dir))
    dataset_path = output_dir / "visual_counterfact_mc.jsonl"
    write_jsonl(dataset_path, rows)
    manifest = {
        "version": VERSION,
        "official_dataset": "mgolov/Visual-Counterfact",
        "official_revision": "f8cd0b2335356b0f01a164df0cac7403f56a6c83",
        "source_parquet_sha256": {
            split: sha256(path) for split, path in source_paths.items()
        },
        "output": str(dataset_path),
        "output_sha256": sha256(dataset_path),
        "audit": audit(rows),
        "selection": "all official color and size rows; no response-based filtering",
        "option_order": (
            "Alternating by immutable official row index, with the offset reversed "
            "between color and size; both image sides share one order."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
