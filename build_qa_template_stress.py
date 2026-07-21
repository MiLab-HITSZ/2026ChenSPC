#!/usr/bin/env python3
"""Build logically equivalent QA templates for answer-direction stress tests."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


VERSION = "qa_template_stress_v2"
VARIANTS = ("cs_first", "cf_first")
TEMPLATES = {
    "contrastive": (
        'For the visual question "{question}", is the correct answer '
        '"{first}" rather than "{second}"?'
    ),
    "instead": (
        'Looking only at the image, should the question "{question}" be answered '
        '"{first}" instead of "{second}"?'
    ),
    "choice": (
        'Between "{first}" and "{second}", does the image support "{first}" '
        'as the answer to "{question}"?'
    ),
    "reject_second": (
        'For the image-based question "{question}", would choosing "{second}" '
        'be incorrect because "{first}" is the right answer?'
    ),
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_options(options: Sequence[Any]) -> List[str]:
    raw = [str(option).strip() for option in options or [] if str(option).strip()]
    if not raw:
        return []
    joined = " \u2022 ".join(raw)
    marker = re.compile(r"(?<![A-Za-z0-9])([A-D])\s*(?:[.)]|:)\s*", re.I)
    matches = list(marker.finditer(joined))
    if len(matches) <= len(raw):
        return raw
    output = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        text = joined[match.end() : end].strip(" \t\r\n\u2022;|")
        output.append(f"{match.group(1).upper()}. {text}")
    return output


def option_map(options: Sequence[Any]) -> Dict[str, str]:
    parsed = {}
    for index, option in enumerate(split_options(options)):
        text = str(option).strip()
        expected = chr(ord("A") + index)
        if text[:1].upper() == expected and len(text) > 1 and text[1] in ".):":
            text = text[2:].strip()
        parsed[expected] = text
    return parsed


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def anchors(item: Mapping[str, Any]) -> Dict[str, str]:
    mc = item.get("multiple_choice") or {}
    options = option_map(list(mc.get("options") or []))
    cf_key = str(mc.get("counterfactual_gt") or "").strip().upper()
    cs_key = str(mc.get("commonsense_gt") or "").strip().upper()
    source = "native_mc_options"
    if cf_key not in options or cs_key not in options or cf_key == cs_key:
        caption = item.get("captioning") or {}
        cf_text = str(caption.get("counterfactual_gt") or "").strip()
        cs_text = str(caption.get("commonsense_gt") or "").strip()
        if not cf_text or not cs_text or cf_text.casefold() == cs_text.casefold():
            raise ValueError(f"invalid answer anchors for {item.get('pair_id')}")
        return {
            "cf_key": "CF",
            "cs_key": "CS",
            "cf_text": cf_text,
            "cs_text": cs_text,
            "source": "caption_gt_fallback_for_invalid_mc_pair",
        }
    return {
        "cf_key": cf_key,
        "cs_key": cs_key,
        "cf_text": options[cf_key],
        "cs_text": options[cs_key],
        "source": source,
    }


def transform(
    item: Mapping[str, Any], template_id: str, variant: str
) -> Dict[str, Any]:
    if template_id not in TEMPLATES:
        raise ValueError(f"unknown template: {template_id}")
    answer = anchors(item)
    if variant == "cs_first":
        first_key, first_text = answer["cs_key"], answer["cs_text"]
        second_key, second_text = answer["cf_key"], answer["cf_text"]
        cf_gt, cs_gt = "no", "yes"
    elif variant == "cf_first":
        first_key, first_text = answer["cf_key"], answer["cf_text"]
        second_key, second_text = answer["cs_key"], answer["cs_text"]
        cf_gt, cs_gt = "yes", "no"
    else:
        raise ValueError(f"unknown variant: {variant}")

    output = json.loads(json.dumps(item))
    source_question = str((output.get("multiple_choice") or {}).get("question") or "").strip()
    if not source_question:
        raise ValueError(f"missing source question for {item.get('pair_id')}")
    question = TEMPLATES[template_id].format(
        question=source_question,
        first=first_text,
        second=second_text,
    )
    output["direct_qa"] = {
        "question": question,
        "counterfactual_gt": cf_gt,
        "commonsense_gt": cs_gt,
    }
    output["qa_template_stress"] = {
        "version": VERSION,
        "template_id": template_id,
        "variant": variant,
        "truth_condition": "first_option_is_correct",
        "anchor_source": answer["source"],
        "source_mc_question": source_question,
        "first_option_key": first_key,
        "first_option_text": first_text,
        "second_option_key": second_key,
        "second_option_text": second_text,
        "counterfactual_gt": cf_gt,
        "commonsense_gt": cs_gt,
    }
    return output


def validate(all_rows: Mapping[str, Mapping[str, List[Dict[str, Any]]]]) -> None:
    template_ids = list(all_rows)
    reference = all_rows[template_ids[0]]
    for template_id, variants in all_rows.items():
        if set(variants) != set(VARIANTS):
            raise AssertionError(f"missing variant for {template_id}")
        if len(variants["cs_first"]) != len(variants["cf_first"]):
            raise AssertionError(f"variant size mismatch for {template_id}")
        for left, right in zip(variants["cs_first"], variants["cf_first"]):
            if left["pair_id"] != right["pair_id"]:
                raise AssertionError(f"pair order mismatch for {template_id}")
            for side in ("counterfactual", "commonsense"):
                labels = {
                    left["direct_qa"][f"{side}_gt"],
                    right["direct_qa"][f"{side}_gt"],
                }
                if labels != {"yes", "no"}:
                    raise AssertionError(
                        f"non-complementary labels at {template_id}/{left['pair_id']}"
                    )
        for variant in VARIANTS:
            for current, expected in zip(variants[variant], reference[variant]):
                if current["pair_id"] != expected["pair_id"]:
                    raise AssertionError("template pair order mismatch")
                current_qa = current["direct_qa"]
                expected_qa = expected["direct_qa"]
                if (
                    current_qa["counterfactual_gt"] != expected_qa["counterfactual_gt"]
                    or current_qa["commonsense_gt"] != expected_qa["commonsense_gt"]
                ):
                    raise AssertionError("ground truth changed across templates")
                lowered = current_qa["question"].casefold()
                if "counterfactual" in lowered or "commonsense" in lowered:
                    raise AssertionError("evaluation-only side label leaked into question")


def label_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    output = {}
    for side in ("counterfactual", "commonsense"):
        output[side] = dict(
            sorted(Counter(row["direct_qa"][f"{side}_gt"] for row in rows).items())
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="CDH-Bench.revised.strict.jsonl")
    parser.add_argument("--output-dir", default="data/qa_template_stress_v2")
    args = parser.parse_args()

    source = Path(args.input)
    output_dir = Path(args.output_dir)
    source_rows = read_jsonl(source)
    all_rows = {
        template_id: {
            variant: [transform(row, template_id, variant) for row in source_rows]
            for variant in VARIANTS
        }
        for template_id in TEMPLATES
    }
    validate(all_rows)

    artifacts = {}
    for template_id, variants in all_rows.items():
        artifacts[template_id] = {}
        for variant, rows in variants.items():
            path = output_dir / template_id / f"{variant}.jsonl"
            write_jsonl(path, rows)
            artifacts[template_id][variant] = {
                "path": str(path),
                "sha256": sha256(path),
                "label_counts": label_counts(rows),
            }

    combined = [
        row
        for variants in all_rows.values()
        for rows in variants.values()
        for row in rows
    ]
    manifest = {
        "version": VERSION,
        "source": str(source),
        "source_sha256": sha256(source),
        "pairs": len(source_rows),
        "templates": {
            key: {
                "surface_form": value,
                "truth_condition": "first_option_is_correct",
                "contains_explicit_negation": key == "reject_second",
            }
            for key, value in TEMPLATES.items()
        },
        "artifacts": artifacts,
        "validation": {
            "pairwise_complementary_labels": True,
            "identical_ground_truth_across_templates": True,
            "evaluation_side_labels_absent_from_questions": True,
            "human_authored_templates": True,
            "semantic_equivalence_basis": (
                "All templates ask whether the first of two mutually exclusive native "
                "answers is correct; cs_first/cf_first only swap those answer texts."
            ),
            "total_rows": len(combined),
        },
        "anchor_source_counts": dict(
            sorted(
                Counter(
                    row["qa_template_stress"]["anchor_source"]
                    for row in all_rows["contrastive"]["cs_first"]
                ).items()
            )
        ),
        "fairness": {
            "method_specific_wording": False,
            "manual_subcategory_rules": False,
            "paired_reference_at_inference": False,
            "single_image_per_example": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
