#!/usr/bin/env python3
"""Collect native and image/no-image candidate scores for paired CDH MC."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cprc_vlm_runtime import (
    candidate_key,
    candidate_logprobs_shared_prefix,
    candidate_vlm_inputs,
    load_model_specs,
    local_candidate_bundle,
)


RowKey = Tuple[str, str]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_pair_ids(path: Path, split: str) -> set[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entry = (manifest.get("split_manifests") or {}).get(split)
    if not isinstance(entry, dict) or not isinstance(entry.get("pair_ids"), list):
        raise ValueError(f"unknown pair split {split!r} in {path}")
    return {str(pair_id) for pair_id in entry["pair_ids"]}


def completed_rows(path: Path) -> set[RowKey]:
    if not path.exists():
        return set()
    return {
        (str(row.get("pair_id")), str(row.get("side")))
        for row in read_jsonl(path)
        if row.get("status") == "ok"
    }


def mc_options(item: Dict[str, Any]) -> List[str]:
    options = list((item.get("multiple_choice") or {}).get("options") or [])
    if not options:
        raise ValueError(f"missing MC options for {item.get('pair_id')}")
    return [str(option).strip() for option in options]


def mc_candidates(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    for option in mc_options(item):
        match = re.match(r"\s*([A-D])(?:\.|\)|\s|$)", option, flags=re.IGNORECASE)
        if match:
            key = match.group(1).upper()
            candidates.append((key, f" {key}"))
    if len(candidates) < 2 or len({key for key, _ in candidates}) != len(candidates):
        raise ValueError(f"invalid MC options for {item.get('pair_id')}")
    return candidates


def mc_prompt(item: Dict[str, Any], final: bool = False) -> str:
    payload = item.get("multiple_choice") or {}
    question = str(payload.get("question") or "").strip()
    body = "\n".join([question, *mc_options(item)])
    body += "\nAnswer with a single letter (A, B, C, or D)."
    return f"{body}\nFinal answer:" if final else body


def image_path(images_root: Path, item: Dict[str, Any], side: str) -> Path:
    subcategory = str(item.get("subcategory") or "").replace(" ", "_").replace("/", "_")
    pair_id = str(item.get("pair_id") or "").replace(" ", "_")
    filename = "counterfactual.png" if side == "counterfactual" else "commonsense.png"
    return images_root / subcategory / pair_id / filename


def generate_native(spec: Any, item: Dict[str, Any], image: Any) -> Tuple[str, int]:
    import torch

    model, processor, family = local_candidate_bundle(spec)
    inputs = candidate_vlm_inputs(model, processor, family, mc_prompt(item), image)
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(spec.max_tokens),
            do_sample=False,
        )
    width = int(inputs["input_ids"].shape[1])
    text = processor.decode(output[0, width:], skip_special_tokens=True).strip()
    return text, int((time.time() - started) * 1000)


def score_candidates(
    spec: Any,
    item: Dict[str, Any],
    image: Any,
) -> Tuple[List[Dict[str, Any]], int]:
    model, processor, family = local_candidate_bundle(spec)
    candidates = mc_candidates(item)
    started = time.time()
    distributions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for view, current_image in (("image", image), ("prior", None)):
        scores = candidate_logprobs_shared_prefix(
            model,
            processor,
            mc_prompt(item, final=True),
            current_image,
            [text for _, text in candidates],
            family=family,
        )
        distributions[view] = {
            key: score for (key, _), score in zip(candidates, scores)
        }

    rows: List[Dict[str, Any]] = []
    for key, text in candidates:
        image_score = distributions["image"][key]
        prior_score = distributions["prior"][key]
        rows.append(
            {
                "key": key,
                "text": text,
                "logp_image": float(image_score["logprob"]),
                "avg_logp_image": float(image_score["avg_logprob"]),
                "logp_prior": float(prior_score["logprob"]),
                "avg_logp_prior": float(prior_score["avg_logprob"]),
                "image_token_count": int(image_score["token_count"]),
                "prior_token_count": int(prior_score["token_count"]),
                "image_token_ids": list(image_score["token_ids"]),
                "prior_token_ids": list(prior_score["token_ids"]),
            }
        )
    return rows, int((time.time() - started) * 1000)


def make_record(
    spec: Any,
    item: Dict[str, Any],
    side: str,
    current_image_path: Path,
    baseline_text: str,
    generation_ms: int,
    candidates: Sequence[Dict[str, Any]],
    score_ms: int,
) -> Dict[str, Any]:
    payload = item.get("multiple_choice") or {}
    baseline = candidate_key(spec, "mc", baseline_text)
    gt = str(payload.get(f"{side}_gt") or "").upper()
    cf_gt = str(payload.get("counterfactual_gt") or "").upper()
    cs_gt = str(payload.get("commonsense_gt") or "").upper()
    image_top = max(candidates, key=lambda row: row["logp_image"])["key"]
    prior_top = max(candidates, key=lambda row: row["logp_prior"])["key"]
    candidate_payload = {
        "backend": "local_transformers_joint_next_token",
        "model": spec.model,
        "model_precision": {
            "quantization": spec.quantization,
            "dtype": spec.dtype,
        },
        "image_preprocessing": "fixed_512",
        "mitigation": "score_collection_only",
        "scoring_prompt": mc_prompt(item, final=True),
        "baseline_pred": baseline_text,
        "baseline_key": baseline,
        "image_top": image_top,
        "prior_top": prior_top,
        "candidates": list(candidates),
    }
    return {
        "model_name": spec.name,
        "backend": spec.backend,
        "model": spec.model,
        "pair_id": str(item.get("pair_id") or ""),
        "cv_group": item.get("cv_group"),
        "category": item.get("category"),
        "subcategory": item.get("subcategory"),
        "task": "mc",
        "side": side,
        "paired_side": "counterfactual" if side == "commonsense" else "commonsense",
        "image_path": str(current_image_path),
        "status": "ok",
        "question": str(payload.get("question") or ""),
        "gt": gt,
        "cf_gt": cf_gt,
        "cs_gt": cs_gt,
        "pred": f"Final answer: {baseline}" if baseline else baseline_text,
        "correct": baseline == gt,
        "commonsense_error": side == "counterfactual" and baseline != gt and baseline == cs_gt,
        "baseline_latency_ms": generation_ms,
        "score_latency_ms": score_ms,
        "cp_vbc": candidate_payload,
        "raw": {
            "baseline_raw": {"text": baseline_text},
            "cp_vbc": candidate_payload,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CDH-Bench.revised.strict.jsonl")
    parser.add_argument("--images-root", default="images")
    parser.add_argument("--pair-manifest", default="runtime/paper/cdh_exposure_ledger.json")
    parser.add_argument("--pair-split", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")

    specs = load_model_specs(args.models)
    if len(specs) != 1:
        raise SystemExit("exactly one model is required")
    spec = specs[0]
    selected_ids = split_pair_ids(Path(args.pair_manifest), args.pair_split)
    items = [
        item
        for item in read_jsonl(Path(args.dataset))
        if str(item.get("pair_id")) in selected_ids
    ]
    if len(items) != len(selected_ids):
        raise ValueError(
            f"pair split contains {len(selected_ids)} ids but dataset resolved {len(items)}"
        )
    tasks = [
        (item, side)
        for item in items
        for side in ("commonsense", "counterfactual")
    ]
    tasks = [
        task
        for index, task in enumerate(tasks)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    output = Path(args.output)
    done = completed_rows(output)
    images_root = Path(args.images_root)
    from PIL import Image

    for position, (item, side) in enumerate(tasks, start=1):
        key = (str(item.get("pair_id")), side)
        if key in done:
            continue
        started = time.time()
        record: Dict[str, Any] = {
            "model_name": spec.name,
            "model": spec.model,
            "pair_id": key[0],
            "task": "mc",
            "side": side,
        }
        error: Optional[Exception] = None
        for _ in range(max(1, args.retry + 1)):
            try:
                current_image_path = image_path(images_root, item, side)
                image = Image.open(current_image_path).convert("RGB")
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                baseline_text, generation_ms = generate_native(spec, item, image)
                candidates, score_ms = score_candidates(spec, item, image)
                record = make_record(
                    spec,
                    item,
                    side,
                    current_image_path,
                    baseline_text,
                    generation_ms,
                    candidates,
                    score_ms,
                )
                error = None
                break
            except Exception as current_error:
                error = current_error
        if error is not None:
            record.update({"status": "error", "error": str(error), "correct": False})
        record["latency_ms"] = int((time.time() - started) * 1000)
        append_jsonl(output, record)
        if position % 25 == 0:
            print(f"{position}/{len(tasks)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
