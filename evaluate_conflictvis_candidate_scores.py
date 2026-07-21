#!/usr/bin/env python3
"""Collect native and image/no-image candidate scores on ConflictVIS QA/MC."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spc_vlm_runtime import (
    candidate_key,
    candidate_vlm_inputs,
    load_model_specs,
    local_candidate_bundle,
)
from evaluate_pope_native import append_jsonl


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["source_id"]) for row in _read_jsonl(str(path)) if row.get("status") == "ok"}


def _prompt(item: Dict[str, Any], final: bool = False) -> str:
    question = str(item["question"]).strip()
    return f"{question}\nFinal answer:" if final else question


def _candidate_tokens(processor: Any, task: str) -> List[Tuple[str, str, int]]:
    keys = ("yes", "no") if task == "qa" else ("A", "B", "C", "D")
    output = []
    for key in keys:
        text = f" {key}"
        token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"candidate {text!r} is not one token: {token_ids}")
        output.append((key, text, int(token_ids[0])))
    return output


def _generate(spec: Any, item: Dict[str, Any], image: Any) -> Tuple[str, int]:
    import torch

    model, processor, family = local_candidate_bundle(spec)
    inputs = candidate_vlm_inputs(model, processor, family, _prompt(item), image)
    started = time.time()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=int(spec.max_tokens), do_sample=False)
    width = int(inputs["input_ids"].shape[1])
    text = processor.decode(generated[0, width:], skip_special_tokens=True).strip()
    return text, int((time.time() - started) * 1000)


def _score(spec: Any, item: Dict[str, Any], image: Any) -> Tuple[List[Dict[str, Any]], int]:
    import torch

    model, processor, family = local_candidate_bundle(spec)
    tokens = _candidate_tokens(processor, str(item["task"]))
    started = time.time()
    distributions: Dict[str, Dict[str, float]] = {}
    for source, current_image in (("image", image), ("prior", None)):
        inputs = candidate_vlm_inputs(model, processor, family, _prompt(item, final=True), current_image)
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, return_dict=True)
            logp = torch.log_softmax(output.logits[0, -1, :].float(), dim=-1)
        distributions[source] = {key: float(logp[token_id].item()) for key, _, token_id in tokens}
    rows = []
    for key, text, token_id in tokens:
        rows.append({
            "key": key,
            "text": text,
            "logp_image": distributions["image"][key],
            "avg_logp_image": distributions["image"][key],
            "logp_prior": distributions["prior"][key],
            "avg_logp_prior": distributions["prior"][key],
            "image_token_count": 1,
            "prior_token_count": 1,
            "image_token_ids": [token_id],
            "prior_token_ids": [token_id],
        })
    return rows, int((time.time() - started) * 1000)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/conflictvis_transfer/conflictvis_candidate_complete.jsonl")
    parser.add_argument("--models", default="configs/qwen_32b_instruct_bf16_fair.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")
    specs = load_model_specs(args.models)
    if len(specs) != 1:
        raise SystemExit("exactly one model is required")
    spec = specs[0]
    selected = [row for index, row in enumerate(_read_jsonl(args.dataset)) if index % args.shard_count == args.shard_index]
    if args.limit > 0:
        selected = selected[: args.limit]
    output = Path(args.output)
    done = _completed(output)
    from PIL import Image

    for position, item in enumerate(selected, start=1):
        if str(item["source_id"]) in done:
            continue
        started = time.time()
        record: Dict[str, Any] = dict(item)
        record.update({"model_name": spec.name, "model": spec.model, "side": "counterfactual"})
        error: Optional[Exception] = None
        for _ in range(max(1, int(args.retry) + 1)):
            try:
                image = Image.open(str(item["image_path"])).convert("RGB")
                baseline_text, generation_ms = _generate(spec, item, image)
                candidates, score_ms = _score(spec, item, image)
                baseline_key = candidate_key(spec, str(item["task"]), baseline_text)
                record.update({
                    "status": "ok",
                    "pred": baseline_key,
                    "baseline_text": baseline_text,
                    "correct": baseline_key == str(item["gt"]),
                    "baseline_latency_ms": generation_ms,
                    "score_latency_ms": score_ms,
                    "cp_vbc": {
                        "backend": "local_transformers_qwen3_vl_joint_next_token",
                        "model": spec.model,
                        "model_precision": {"quantization": spec.quantization, "dtype": spec.dtype},
                        "image_preprocessing": "model_native",
                        "mitigation": "score_collection_only",
                        "scoring_prompt": _prompt(item, final=True),
                        "baseline_pred": baseline_text,
                        "baseline_key": baseline_key,
                        "image_top": max(candidates, key=lambda row: row["logp_image"])["key"],
                        "prior_top": max(candidates, key=lambda row: row["logp_prior"])["key"],
                        "candidates": candidates,
                    },
                })
                error = None
                break
            except Exception as current_error:
                error = current_error
        if error is not None:
            record.update({"status": "error", "error": str(error), "correct": False})
        record["latency_ms"] = int((time.time() - started) * 1000)
        append_jsonl(output, record)
        if position % 25 == 0:
            print(f"shard {args.shard_index}: {position}/{len(selected)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
