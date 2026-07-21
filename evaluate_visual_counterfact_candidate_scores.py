#!/usr/bin/env python3
"""Collect native and image/no-image candidate scores on Visual CounterFact."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spc_vlm_runtime import (
    candidate_key,
    candidate_logprobs_shared_prefix as _candidate_logprobs_shared_prefix,
    candidate_vlm_inputs,
    load_model_specs,
    local_candidate_bundle,
)
from evaluate_pope_native import append_jsonl


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["source_id"])
        for row in read_jsonl(path)
        if row.get("status") == "ok"
    }


def prompt(item: Dict[str, Any], final: bool = False) -> str:
    text = str(item["question"]).strip()
    return f"{text}\nFinal answer:" if final else text


def generate(spec: Any, item: Dict[str, Any], image: Any) -> Tuple[str, int]:
    import torch

    model, processor, family = local_candidate_bundle(spec)
    inputs = candidate_vlm_inputs(model, processor, family, prompt(item), image)
    started = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=int(spec.max_tokens), do_sample=False
        )
    width = int(inputs["input_ids"].shape[1])
    text = processor.decode(generated[0, width:], skip_special_tokens=True).strip()
    return text, int((time.time() - started) * 1000)


def score(spec: Any, item: Dict[str, Any], image: Any) -> Tuple[List[Dict[str, Any]], int]:
    model, processor, family = local_candidate_bundle(spec)
    candidates = [(str(key), f" {key}") for key in item["candidate_keys"]]
    started = time.time()
    distributions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source, current_image in (("image", image), ("prior", None)):
        scored = _candidate_logprobs_shared_prefix(
            model,
            processor,
            prompt(item, final=True),
            current_image,
            [text for _, text in candidates],
            family=family,
        )
        distributions[source] = {
            key: values for (key, _), values in zip(candidates, scored)
        }
    rows = []
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/visual_counterfact_frozen_v1/visual_counterfact_mc.jsonl",
    )
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
    source = read_jsonl(Path(args.dataset))
    selected = [
        row
        for index, row in enumerate(source)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    output = Path(args.output)
    done = completed(output)
    from PIL import Image

    for position, item in enumerate(selected, start=1):
        if str(item["source_id"]) in done:
            continue
        started = time.time()
        record: Dict[str, Any] = dict(item)
        record.update({"model_name": spec.name, "model": spec.model})
        error: Optional[Exception] = None
        for _ in range(max(1, int(args.retry) + 1)):
            try:
                image = Image.open(str(item["image_path"])).convert("RGB")
                baseline_text, generation_ms = generate(spec, item, image)
                candidates, score_ms = score(spec, item, image)
                baseline_key = candidate_key(spec, "mc", baseline_text)
                record.update(
                    {
                        "status": "ok",
                        "pred": baseline_key,
                        "baseline_text": baseline_text,
                        "correct": baseline_key == str(item["gt"]),
                        "baseline_latency_ms": generation_ms,
                        "score_latency_ms": score_ms,
                        "cp_vbc": {
                            "backend": "local_transformers_joint_next_token",
                            "model": spec.model,
                            "model_precision": {
                                "quantization": spec.quantization,
                                "dtype": spec.dtype,
                            },
                            "image_preprocessing": "model_native",
                            "mitigation": "score_collection_only",
                            "scoring_prompt": prompt(item, final=True),
                            "baseline_pred": baseline_text,
                            "baseline_key": baseline_key,
                            "image_top": max(
                                candidates, key=lambda row: row["logp_image"]
                            )["key"],
                            "prior_top": max(
                                candidates, key=lambda row: row["logp_prior"]
                            )["key"],
                            "candidates": candidates,
                        },
                    }
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
            print(
                f"shard {args.shard_index}: {position}/{len(selected)}", flush=True
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
