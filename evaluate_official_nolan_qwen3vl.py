#!/usr/bin/env python3
"""Evaluate the official token-level NoLan rule on supported CDH VLMs."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from spc_vlm_runtime import (
    ModelSpec,
    candidate_vlm_inputs,
    load_model_specs,
    local_candidate_bundle,
)
from official_nolan_qwen3vl import OFFICIAL_NOLAN_BETA, generate_with_nolan


Identity = Tuple[str, str, str]


def row_identity(row: Mapping[str, Any]) -> Identity:
    return (
        str(row.get("pair_id") or ""),
        str(row.get("task") or ""),
        str(row.get("side") or ""),
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def completed_identities(path: Path) -> set[Identity]:
    if not path.exists():
        return set()
    return {
        row_identity(row)
        for row in load_jsonl(path)
        if str(row.get("status") or "") == "ok"
    }


def scoring_prompt(row: Mapping[str, Any]) -> str:
    blocks = [
        row.get("cp_vbc"),
        (row.get("raw") or {}).get("cp_vbc") if isinstance(row.get("raw"), dict) else None,
    ]
    for block in blocks:
        if isinstance(block, dict) and block.get("scoring_prompt"):
            return str(block["scoring_prompt"])
    question = str(row.get("question") or "").strip()
    if str(row.get("task")) == "qa":
        return f"{question}\nAnswer with yes or no.\nFinal answer:"
    raise ValueError(f"missing MC scoring prompt for {row_identity(row)}")


def source_native_answer(row: Mapping[str, Any]) -> Tuple[str, Optional[str], bool]:
    blocks = [
        row.get("cp_vbc"),
        (row.get("raw") or {}).get("cp_vbc") if isinstance(row.get("raw"), dict) else None,
    ]
    task = str(row.get("task") or "")
    gt = str(row.get("gt") or "")
    normalized_gt = gt.lower() if task == "qa" else gt.upper()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        prediction = str(block.get("baseline_pred") or "")
        key = block.get("baseline_key")
        if key is not None:
            normalized_key = str(key).lower() if task == "qa" else str(key).upper()
            return prediction, normalized_key, normalized_key == normalized_gt
    prediction = str(row.get("pred") or "")
    key = generated_answer_key(task, prediction)
    return prediction, key, key == normalized_gt


def answer_stop_token_ids(
    row: Mapping[str, Any], processor: Optional[Any] = None
) -> List[int]:
    blocks = [
        row.get("cp_vbc"),
        (row.get("raw") or {}).get("cp_vbc")
        if isinstance(row.get("raw"), dict)
        else None,
    ]
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("candidates"), list):
            continue
        values: List[int] = []
        for candidate in block["candidates"]:
            token_ids = candidate.get("image_token_ids")
            if isinstance(token_ids, list) and token_ids:
                values.append(int(token_ids[-1]))
            if processor is not None:
                tokenizer = getattr(processor, "tokenizer", processor)
                key_text = str(candidate.get("key") or "")
                candidate_text = str(candidate.get("text") or "").strip()
                variants = {
                    key_text,
                    key_text.lower(),
                    key_text.upper(),
                    key_text.capitalize(),
                    candidate_text,
                    candidate_text.lower(),
                    candidate_text.upper(),
                    candidate_text.capitalize(),
                }
                for text in variants:
                    if not text:
                        continue
                    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
                    if encoded and isinstance(encoded[0], list):
                        encoded = encoded[0]
                    if encoded:
                        values.append(int(encoded[-1]))
        if values:
            return sorted(set(values))
    return []


def generated_answer_key(task: str, text: str) -> Optional[str]:
    """Parse an explicit generated answer without matching prose articles."""
    value = str(text or "").strip()
    if task == "mc":
        patterns = (
            r"^\s*([A-D])(?:\s|[.):,-]|$)",
            r"(?:final\s+answer|answer)\s*(?:is|:|-)?\s*([A-D])\b",
            r"(?:^|\n)\s*([A-D])\s*[.)]?\s*$",
        )
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        return None
    if task == "qa":
        patterns = (
            r"^\s*(yes|no)\b",
            r"(?:final\s+answer|answer)\s*(?:is|:|-)?\s*(yes|no)\b",
            r"(?:^|\n)\s*(yes|no)\s*[.!]?\s*$",
        )
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).lower()
        return None
    raise ValueError(f"unknown task: {task}")


def select_rows(
    rows: Sequence[Dict[str, Any]],
    tasks: set[str],
    shard_index: int,
    shard_count: int,
    limit: int,
) -> List[Dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("status") or "") == "ok"
        and str(row.get("task") or "") in tasks
    ]
    selected.sort(key=row_identity)
    selected = [
        row for index, row in enumerate(selected) if index % shard_count == shard_index
    ]
    return selected[:limit] if limit > 0 else selected


def summarize(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status") or "") == "ok":
            groups[(str(row["task"]), str(row["side"]))].append(row)
    payload: Dict[str, Any] = {}
    for (task, side), values in sorted(groups.items()):
        native = sum(bool(row.get("native_correct")) for row in values) / len(values)
        nolan = sum(bool(row.get("correct")) for row in values) / len(values)
        payload[f"{task}/{side}"] = {
            "n": len(values),
            "native_accuracy": native,
            "nolan_accuracy": nolan,
            "delta_accuracy_points": 100.0 * (nolan - native),
            "changed": sum(row.get("native_key") != row.get("pred_key") for row in values),
            "repairs": sum(
                not bool(row.get("native_correct")) and bool(row.get("correct"))
                for row in values
            ),
            "harms": sum(
                bool(row.get("native_correct")) and not bool(row.get("correct"))
                for row in values
            ),
        }
    return payload


def refresh_native_metadata(
    output: Path,
    source_rows: Sequence[Mapping[str, Any]],
    native_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    source_by_identity = {row_identity(row): row for row in source_rows}
    native_by_identity = (
        {row_identity(row): row for row in native_rows} if native_rows is not None else {}
    )
    rows = load_jsonl(output) if output.exists() else []
    for row in rows:
        source = source_by_identity.get(row_identity(row))
        if source is None or str(row.get("status") or "") != "ok":
            continue
        candidate_pred, candidate_key, candidate_correct = source_native_answer(source)
        row["candidate_native_pred"] = candidate_pred
        row["candidate_native_key"] = candidate_key
        row["candidate_native_correct"] = candidate_correct
        matched_native = native_by_identity.get(row_identity(row))
        if matched_native is not None and str(matched_native.get("status") or "") == "ok":
            native_pred = str(matched_native.get("pred") or "")
            native_key = matched_native.get("pred_key")
            native_correct = bool(matched_native.get("correct"))
        else:
            native_pred, native_key, native_correct = (
                candidate_pred,
                candidate_key,
                candidate_correct,
            )
        row["native_pred"] = native_pred
        row["native_key"] = native_key
        row["native_correct"] = native_correct
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(output)
    return rows


def evaluate_row(
    spec: ModelSpec,
    model: Any,
    processor: Any,
    family: str,
    row: Mapping[str, Any],
    *,
    beta: float,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    trace_steps: int,
) -> Dict[str, Any]:
    if family not in {"qwen3_vl", "llava_next"}:
        raise ValueError(f"unsupported family for the official NoLan port: {family}")
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    prompt = scoring_prompt(row)
    multimodal = candidate_vlm_inputs(model, processor, family, prompt, image)
    text_only = candidate_vlm_inputs(model, processor, family, prompt, None)
    started = time.time()
    generation = generate_with_nolan(
        model,
        processor,
        multimodal,
        text_only,
        beta=beta,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        trace_steps=trace_steps,
        stop_token_ids=answer_stop_token_ids(row, processor),
    )
    task = str(row["task"])
    gt = str(row["gt"])
    native_pred, native_key, native_correct = source_native_answer(row)
    pred_key = generated_answer_key(task, generation.text)
    normalized_gt = gt.lower() if task == "qa" else gt.upper()
    return {
        "pair_id": row.get("pair_id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "task": task,
        "side": row.get("side"),
        "image_path": row.get("image_path"),
        "question": row.get("question"),
        "gt": gt,
        "status": "ok",
        "model_name": spec.name,
        "model": spec.model,
        "prompt": prompt,
        "native_pred": native_pred,
        "native_key": native_key,
        "native_correct": native_correct,
        "pred": generation.text,
        "pred_key": pred_key,
        "correct": pred_key == normalized_gt,
        "latency_ms": int((time.time() - started) * 1000),
        "nolan": generation.diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models", default="configs/qwen_32b_instruct_nolan_official.json"
    )
    parser.add_argument("--beta", type=float, default=OFFICIAL_NOLAN_BETA)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--tasks", nargs="+", choices=("mc", "qa"), default=("mc", "qa"))
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trace-steps", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--native-output",
        help="Matched beta=0 decoding output used as the native reference.",
    )
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
    rows = select_rows(
        source_rows,
        set(args.tasks),
        args.shard_index,
        args.shard_count,
        args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    native_rows = load_jsonl(Path(args.native_output)) if args.native_output else None
    if args.summary_only:
        all_results = refresh_native_metadata(output, source_rows, native_rows)
        summary = {
            "input": str(Path(args.input)),
            "output": str(output),
            "native_output": str(Path(args.native_output)) if args.native_output else None,
            "model": spec.name,
            "beta": float(args.beta),
            "max_new_tokens": int(args.max_new_tokens),
            "decoding": "sampling" if args.do_sample else "greedy",
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "metrics": summarize(all_results),
        }
        summary_path = output.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2), flush=True)
        return
    done = set() if args.no_resume else completed_identities(output)
    model, processor, family = local_candidate_bundle(spec)

    output_mode = "w" if args.no_resume else "a"
    with output.open(output_mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
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
                    beta=args.beta,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    trace_steps=args.trace_steps,
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
            print(
                f"[{index}/{len(rows)}] {identity}: "
                f"{result['status']} {result.get('native_key')}->{result.get('pred_key')}",
                flush=True,
            )

    all_results = refresh_native_metadata(output, source_rows, native_rows)
    summary = {
        "input": str(Path(args.input)),
        "output": str(output),
        "native_output": str(Path(args.native_output)) if args.native_output else None,
        "model": spec.name,
        "beta": float(args.beta),
        "max_new_tokens": int(args.max_new_tokens),
        "decoding": "sampling" if args.do_sample else "greedy",
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "metrics": summarize(all_results),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
