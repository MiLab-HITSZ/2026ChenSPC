#!/usr/bin/env python3
"""Materialize the candidate-complete ConflictVIS tracks for frozen transfer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_options(question: str) -> List[str]:
    matches = re.findall(
        r"\(([A-D])\)\s*(.*?)(?=\s*\([A-D]\)|\s*Answer with|$)",
        question,
        flags=re.IGNORECASE | re.DOTALL,
    )
    options = [f"{letter.upper()}. {' '.join(text.split())}" for letter, text in matches]
    if len(options) != 4:
        raise ValueError(f"expected four options, got {options!r}")
    return options


def _image_digest(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="xiaoyuanliu/conflict_vis")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/conflictvis_transfer")
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset = load_dataset(args.dataset, split=args.split)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    concept_digests: Dict[int, str] = {}
    for source in dataset:
        question_type = str(source["question_type"]).upper()
        if question_type not in {"YN", "MC"}:
            continue
        source_id = str(source["id"])
        concept_id = (int(source_id) - 1) // 3 + 1
        image = source["images"].convert("RGB")
        digest = _image_digest(image)
        previous = concept_digests.setdefault(concept_id, digest)
        if previous != digest:
            raise ValueError(f"question variants use different images for concept {concept_id}")
        image_path = images_dir / f"concept_{concept_id:04d}.png"
        if not image_path.exists():
            image.save(image_path, format="PNG")
        task = "qa" if question_type == "YN" else "mc"
        answer = str(source["answer"]).strip().lower() if task == "qa" else str(source["answer"]).strip().upper()
        if task == "qa" and answer not in {"yes", "no"}:
            raise ValueError(f"invalid binary answer: {answer!r}")
        if task == "mc" and answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"invalid MC answer: {answer!r}")
        row: Dict[str, Any] = {
            "source_id": source_id,
            "pair_id": f"ConflictVIS-{concept_id:04d}",
            "concept_id": concept_id,
            "task": task,
            "question": str(source["question"]).strip(),
            "gt": answer,
            "conflict_target": str(source["conflict_target"]),
            "image_path": str(image_path),
            "image_sha256": digest,
        }
        if task == "mc":
            row["options"] = _parse_options(row["question"])
        rows.append(row)

    dataset_path = output_dir / "conflictvis_candidate_complete.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    revision = HfApi().dataset_info(args.dataset).sha
    manifest = {
        "version": "conflictvis_transfer_v1",
        "source": args.dataset,
        "source_revision": revision,
        "split": args.split,
        "selection": "All official YN and MC rows; open-ended SUBJ rows excluded before inference.",
        "response_blind_selection": True,
        "rows": len(rows),
        "qa_rows": sum(row["task"] == "qa" for row in rows),
        "mc_rows": sum(row["task"] == "mc" for row in rows),
        "concepts": len({row["concept_id"] for row in rows}),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "official_repository": "https://github.com/xyliu-cs/ConflictVIS",
        "official_repository_commit": "34567635933a3be59ff03fa13f3991da33a7b4f8",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
