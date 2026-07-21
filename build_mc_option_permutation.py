#!/usr/bin/env python3
"""Build label-balanced MC option-order interventions without changing semantics."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from cdh_caption_claims import split_multiple_choice_options


VERSION = "cdh_mc_option_permutation_v1"
LETTERS = "ABCD"
PERMUTATIONS = {
    "rotate_1": (1, 2, 3, 0),
    "rotate_2": (2, 3, 0, 1),
    "reverse": (3, 2, 1, 0),
}


def option_text(option: str) -> str:
    match = re.match(r"\s*[A-D](?:\.|\)|:)\s*(.*)\s*$", str(option), flags=re.I)
    if not match:
        raise ValueError(f"option lacks an A-D prefix: {option!r}")
    return match.group(1).strip()


def canonical_options(raw_options: Any) -> List[str]:
    """Merge comma-split continuations back into their labelled option."""
    output: List[str] = []
    for value in split_multiple_choice_options(raw_options or []):
        text = str(value).strip()
        if re.match(r"^[A-D](?:\.|\)|:)\s*", text, flags=re.I):
            output.append(text)
        elif output:
            output[-1] = f"{output[-1]}, {text}"
        else:
            raise ValueError(f"orphan option continuation: {text!r}")
    return output


def permute_item(item: Mapping[str, Any], permutation: Sequence[int], name: str) -> Dict[str, Any]:
    if sorted(int(value) for value in permutation) != list(range(4)):
        raise ValueError("permutation must contain every option index exactly once")
    output = copy.deepcopy(dict(item))
    multiple_choice = output.get("multiple_choice") or {}
    options = canonical_options(multiple_choice.get("options") or [])
    if len(options) != 4:
        raise ValueError(f"{item.get('pair_id')}: expected four options, found {len(options)}")
    texts = [option_text(option) for option in options]
    old_to_new = {old: new for new, old in enumerate(permutation)}
    multiple_choice["options"] = [
        f"{LETTERS[new]}. {texts[old]}" for new, old in enumerate(permutation)
    ]
    for field in ("counterfactual_gt", "commonsense_gt"):
        old_letter = str(multiple_choice.get(field, "")).strip().upper()
        if old_letter not in LETTERS:
            raise ValueError(f"{item.get('pair_id')}: invalid {field} {old_letter!r}")
        multiple_choice[field] = LETTERS[old_to_new[LETTERS.index(old_letter)]]
    output["multiple_choice"] = multiple_choice
    output["mc_option_permutation"] = {
        "version": VERSION,
        "name": name,
        "new_position_to_old_position": [int(value) for value in permutation],
        "old_letter_to_new_letter": {
            LETTERS[old]: LETTERS[new] for old, new in sorted(old_to_new.items())
        },
    }
    return output


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_semantics(
    originals: Sequence[Mapping[str, Any]], variants: Sequence[Mapping[str, Any]]
) -> None:
    if len(originals) != len(variants):
        raise AssertionError("row count changed")
    for original, variant in zip(originals, variants):
        for field in ("pair_id", "category", "subcategory", "direct_qa", "captioning"):
            if original.get(field) != variant.get(field):
                raise AssertionError(f"{original.get('pair_id')}: changed field {field}")
        old_mc = original.get("multiple_choice") or {}
        new_mc = variant.get("multiple_choice") or {}
        if old_mc.get("question") != new_mc.get("question"):
            raise AssertionError(f"{original.get('pair_id')}: MC question changed")
        old_texts = sorted(
            option_text(value)
            for value in canonical_options(old_mc.get("options") or [])
        )
        new_texts = sorted(option_text(value) for value in new_mc.get("options") or [])
        if old_texts != new_texts:
            raise AssertionError(f"{original.get('pair_id')}: option semantics changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="CDH-Bench.revised.strict.jsonl")
    parser.add_argument("--output-dir", default="data/mc_option_permutation_v1")
    args = parser.parse_args()

    source = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = read_jsonl(source)
    artifacts = {}
    for name, permutation in PERMUTATIONS.items():
        variants = [permute_item(item, permutation, name) for item in rows]
        verify_semantics(rows, variants)
        destination = output_dir / f"{name}.jsonl"
        write_jsonl(destination, variants)
        artifacts[name] = {
            "path": str(destination),
            "rows": len(variants),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "permutation": list(permutation),
        }

    manifest = {
        "version": VERSION,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "artifacts": artifacts,
        "semantic_contract": (
            "Only option order, A-D prefixes, and the corresponding answer letters "
            "change. Image, question, option texts, pair membership, and both visual "
            "sides remain identical."
        ),
        "manifest_sha256": canonical_hash(artifacts),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
