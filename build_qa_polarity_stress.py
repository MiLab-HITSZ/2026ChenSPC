#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from cdh_caption_claims import split_multiple_choice_options


VERSION = "qa_polarity_stress_v1"
VARIANTS = ("cs_first", "cf_first")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option_map(options: List[Any]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for index, option in enumerate(split_multiple_choice_options(options)):
        text = str(option).strip()
        expected = chr(ord("A") + index)
        prefix = text[:1].upper()
        if prefix == expected and len(text) > 1 and text[1] in ".):":
            text = text[2:].strip()
        parsed[expected] = text
    return parsed


def build_question(source_question: str, first: str, second: str) -> str:
    return (
        f'For the visual question "{source_question.strip()}", is the correct '
        f'answer "{first}" rather than "{second}"?'
    )


def transform(item: Dict[str, Any], variant: str) -> Dict[str, Any]:
    output = json.loads(json.dumps(item))
    mc = output.get("multiple_choice") or {}
    options = option_map(list(mc.get("options") or []))
    cf_key = str(mc.get("counterfactual_gt") or "").strip().upper()
    cs_key = str(mc.get("commonsense_gt") or "").strip().upper()
    anchor_source = "native_mc_options"
    if cf_key not in options or cs_key not in options or cf_key == cs_key:
        caption = output.get("captioning") or {}
        cf_text = str(caption.get("counterfactual_gt") or "").strip()
        cs_text = str(caption.get("commonsense_gt") or "").strip()
        if not cf_text or not cs_text or cf_text.casefold() == cs_text.casefold():
            raise ValueError(f"invalid polarity anchors for {item.get('pair_id')}")
        options = {"CF": cf_text, "CS": cs_text}
        cf_key, cs_key = "CF", "CS"
        anchor_source = "caption_gt_fallback_for_invalid_mc_pair"

    if variant == "cs_first":
        first_key, second_key = cs_key, cf_key
        cf_gt, cs_gt = "no", "yes"
    elif variant == "cf_first":
        first_key, second_key = cf_key, cs_key
        cf_gt, cs_gt = "yes", "no"
    else:
        raise ValueError(f"unknown variant: {variant}")

    source_question = str(mc.get("question") or "").strip()
    if not source_question:
        raise ValueError(f"missing MC question for {item.get('pair_id')}")
    output["direct_qa"] = {
        "question": build_question(
            source_question, options[first_key], options[second_key]
        ),
        "counterfactual_gt": cf_gt,
        "commonsense_gt": cs_gt,
    }
    output["qa_polarity_stress"] = {
        "version": VERSION,
        "variant": variant,
        "anchor_source": anchor_source,
        "source_mc_question": source_question,
        "first_option_key": first_key,
        "first_option_text": options[first_key],
        "second_option_key": second_key,
        "second_option_text": options[second_key],
        "counterfactual_gt": cf_gt,
        "commonsense_gt": cs_gt,
    }
    return output


def label_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts = {"counterfactual": Counter(), "commonsense": Counter()}
    for row in rows:
        qa = row["direct_qa"]
        counts["counterfactual"][qa["counterfactual_gt"]] += 1
        counts["commonsense"][qa["commonsense_gt"]] += 1
    return {side: dict(sorted(values.items())) for side, values in counts.items()}


def validate_pairwise_complements(
    cs_first: List[Dict[str, Any]], cf_first: List[Dict[str, Any]]
) -> None:
    if len(cs_first) != len(cf_first):
        raise AssertionError("variant sizes differ")
    for left, right in zip(cs_first, cf_first):
        if left["pair_id"] != right["pair_id"]:
            raise AssertionError("variant pair order differs")
        for side in ("counterfactual", "commonsense"):
            first = left["direct_qa"][f"{side}_gt"]
            second = right["direct_qa"][f"{side}_gt"]
            if {first, second} != {"yes", "no"}:
                raise AssertionError(f"non-complementary labels at {left['pair_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build answer-direction-balanced QA variants from native MC anchors."
    )
    parser.add_argument("--input", default="CDH-Bench.revised.strict.jsonl")
    parser.add_argument("--output-dir", default="data/qa_polarity_stress_v1")
    args = parser.parse_args()

    source = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = read_jsonl(source)
    variants = {
        variant: [transform(item, variant) for item in rows]
        for variant in VARIANTS
    }
    validate_pairwise_complements(variants["cs_first"], variants["cf_first"])

    paths: Dict[str, Path] = {}
    for variant, transformed in variants.items():
        path = output_dir / f"{variant}.jsonl"
        write_jsonl(path, transformed)
        paths[variant] = path

    combined_counts = label_counts(variants["cs_first"] + variants["cf_first"])
    expected = {"yes": len(rows), "no": len(rows)}
    if any(counts != expected for counts in combined_counts.values()):
        raise AssertionError(f"combined labels are not balanced: {combined_counts}")

    manifest = {
        "version": VERSION,
        "source": str(source),
        "source_sha256": sha256(source),
        "pairs": len(rows),
        "construction": (
            "For each native MC question, ask whether its CF answer is correct rather "
            "than its CS answer, and vice versa. The two variants use the same image, "
            "source question, and answer texts, but reverse the yes/no ground truth. "
            "If the native MC annotation does not distinguish CF and CS, the paired "
            "caption ground truths supply the two answer texts."
        ),
        "variants": {
            variant: {
                "path": str(path),
                "sha256": sha256(path),
                "label_counts": label_counts(variants[variant]),
            }
            for variant, path in paths.items()
        },
        "combined_label_counts": combined_counts,
        "anchor_source_counts": dict(
            sorted(
                Counter(
                    row["qa_polarity_stress"]["anchor_source"]
                    for row in variants["cs_first"]
                ).items()
            )
        ),
        "fairness": {
            "paired_reference_at_inference": False,
            "method_specific_wording": False,
            "semantic_templates": 1,
            "manual_subcategory_rules": False,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
