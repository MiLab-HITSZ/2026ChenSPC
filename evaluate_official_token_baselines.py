#!/usr/bin/env python3
"""Evaluate released token-level hallucination baselines on CDH-Bench."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image

from spc_vlm_runtime import (
    ModelSpec,
    candidate_vlm_inputs,
    load_model_specs,
    local_candidate_bundle,
)
from evaluate_official_nolan_qwen3vl import (
    answer_stop_token_ids,
    completed_identities,
    generated_answer_key,
    load_jsonl,
    row_identity,
    scoring_prompt,
    select_rows,
    source_native_answer,
)
from official_token_baselines import (
    build_mfcd_images,
    generate_native,
    generate_with_mfcd,
    generate_with_pai,
    generate_with_vcd,
)


METHODS = ("vcd", "mfcd", "pai")


def _normalized_key(task: str, value: str) -> str:
    return str(value).lower() if task == "qa" else str(value).upper()


def evaluate_row(
    spec: ModelSpec,
    model: Any,
    processor: Any,
    family: str,
    row: Mapping[str, Any],
    methods: Sequence[str],
    max_new_tokens: int,
    trace_steps: int,
) -> Dict[str, Any]:
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    prompt = scoring_prompt(row)
    task = str(row["task"])
    multimodal = candidate_vlm_inputs(model, processor, family, prompt, image)
    text_only = None
    stop_ids = answer_stop_token_ids(row, processor)

    gt = _normalized_key(task, str(row["gt"]))
    saved_pred, saved_key, saved_correct = source_native_answer(row)
    native_generation = generate_native(
        model,
        processor,
        multimodal,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_ids,
        trace_steps=trace_steps,
    )
    native_parsed = generated_answer_key(task, native_generation.text)
    native_key = (
        _normalized_key(task, native_parsed)
        if native_parsed is not None
        else None
    )
    native_correct = native_key == gt
    outputs: Dict[str, Any] = {}
    for method in methods:
        started = time.time()
        if method == "vcd":
            generation = generate_with_vcd(
                model,
                processor,
                multimodal,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_ids,
                trace_steps=trace_steps,
            )
        elif method == "mfcd":
            high_image, low_image = build_mfcd_images(image)
            high_inputs = candidate_vlm_inputs(
                model, processor, family, prompt, high_image
            )
            low_inputs = candidate_vlm_inputs(
                model, processor, family, prompt, low_image
            )
            generation = generate_with_mfcd(
                model,
                processor,
                multimodal,
                high_inputs,
                low_inputs,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_ids,
                trace_steps=trace_steps,
            )
        elif method == "pai":
            if text_only is None:
                text_only = candidate_vlm_inputs(
                    model, processor, family, prompt, None
                )
            generation = generate_with_pai(
                model,
                processor,
                multimodal,
                text_only,
                family,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_ids,
                trace_steps=trace_steps,
            )
        else:
            raise ValueError(f"unknown baseline: {method}")

        parsed = generated_answer_key(task, generation.text)
        pred_key = _normalized_key(task, parsed) if parsed is not None else None
        outputs[method] = {
            "pred": generation.text,
            "pred_key": pred_key,
            "correct": pred_key == gt,
            "latency_ms": int((time.time() - started) * 1000),
            "token_ids": generation.token_ids,
            **generation.diagnostics,
        }

    return {
        "pair_id": row.get("pair_id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "task": task,
        "side": row.get("side"),
        "image_path": row.get("image_path"),
        "question": row.get("question"),
        "gt": str(row["gt"]),
        "status": "ok",
        "model_name": spec.name,
        "model": spec.model,
        "prompt": prompt,
        "saved_native_pred": saved_pred,
        "saved_native_key": saved_key,
        "saved_native_correct": saved_correct,
        "native_pred": native_generation.text,
        "native_key": native_key,
        "native_correct": native_correct,
        "native_token_ids": native_generation.token_ids,
        "native_decoder": native_generation.diagnostics,
        "baselines": outputs,
    }


def summarize(
    rows: Iterable[Mapping[str, Any]], methods: Sequence[str]
) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status") or "") == "ok":
            groups[(str(row["task"]), str(row["side"]))].append(row)

    result: Dict[str, Any] = {}
    for (task, side), values in sorted(groups.items()):
        group: Dict[str, Any] = {
            "n": len(values),
            "native_accuracy": (
                sum(bool(row["native_correct"]) for row in values) / len(values)
            ),
            "native_parse_failures": sum(
                row.get("native_key") is None for row in values
            ),
        }
        for method in methods:
            available = [
                row
                for row in values
                if isinstance(row.get("baselines"), dict)
                and isinstance(row["baselines"].get(method), dict)
            ]
            if not available:
                continue
            accuracy = (
                sum(bool(row["baselines"][method]["correct"]) for row in available)
                / len(available)
            )
            native_accuracy = (
                sum(bool(row["native_correct"]) for row in available) / len(available)
            )
            group[method] = {
                "n": len(available),
                "accuracy": accuracy,
                "delta_accuracy_points": 100.0 * (accuracy - native_accuracy),
                "changed": sum(
                    str(row["native_key"])
                    != str(row["baselines"][method]["pred_key"])
                    for row in available
                ),
                "repairs": sum(
                    not bool(row["native_correct"])
                    and bool(row["baselines"][method]["correct"])
                    for row in available
                ),
                "harms": sum(
                    bool(row["native_correct"])
                    and not bool(row["baselines"][method]["correct"])
                    for row in available
                ),
                "parse_failures": sum(
                    row["baselines"][method].get("pred_key") is None
                    for row in available
                ),
            }
        result[f"{task}/{side}"] = group
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=("mc", "qa"), default=("mc", "qa")
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trace-steps", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count)")
    specs = load_model_specs(args.models)
    if len(specs) != 1:
        raise ValueError("the evaluator requires exactly one model")
    spec = specs[0]
    source_rows = load_jsonl(Path(args.input))
    selected = select_rows(
        source_rows,
        set(args.tasks),
        args.shard_index,
        args.shard_count,
        args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        results = load_jsonl(output)
    else:
        done = set() if args.no_resume else completed_identities(output)
        model, processor, family = local_candidate_bundle(spec)
        random.seed(42)
        try:
            import numpy as np

            np.random.seed(42)
        except ImportError:
            pass
        import torch

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        mode = "w" if args.no_resume else "a"
        with output.open(mode, encoding="utf-8") as handle:
            for index, row in enumerate(selected, start=1):
                identity = row_identity(row)
                if identity in done:
                    continue
                try:
                    result = evaluate_row(
                        spec,
                        model,
                        processor,
                        family,
                        row,
                        args.methods,
                        args.max_new_tokens,
                        args.trace_steps,
                    )
                except Exception as error:
                    if args.fail_fast:
                        raise
                    result = {
                        "pair_id": row.get("pair_id"),
                        "task": row.get("task"),
                        "side": row.get("side"),
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                handle.flush()
                compact = {
                    method: (
                        result.get("baselines", {})
                        .get(method, {})
                        .get("pred_key")
                    )
                    for method in args.methods
                }
                print(
                    f"[{index}/{len(selected)}] {identity}: "
                    f"{result['status']} {compact}",
                    flush=True,
                )
        results = load_jsonl(output)

    summary = {
        "input": str(Path(args.input)),
        "output": str(output),
        "model": spec.name,
        "methods": list(args.methods),
        "protocol": "released_full_vocabulary_token_level_generation",
        "max_new_tokens": int(args.max_new_tokens),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "metrics": summarize(results, args.methods),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
