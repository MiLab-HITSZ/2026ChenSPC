#!/usr/bin/env python3
"""Evaluate official tokenwise NoLan under the task's native candidate set."""

from __future__ import annotations

import argparse
import json
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
from evaluate_official_nolan_qwen3vl import (
    completed_identities,
    load_jsonl,
    row_identity,
    scoring_prompt,
    select_rows,
    source_native_answer,
)
from official_nolan_qwen3vl import (
    OFFICIAL_NOLAN_BETA,
    OFFICIAL_NOLAN_COMMIT,
    OFFICIAL_NOLAN_REPOSITORY,
    choose_candidate_with_nolan,
)


def candidate_specs(row: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    blocks = [
        row.get("cp_vbc"),
        (row.get("raw") or {}).get("cp_vbc")
        if isinstance(row.get("raw"), dict)
        else None,
    ]
    for block in blocks:
        if isinstance(block, dict) and isinstance(block.get("candidates"), list):
            values = block["candidates"]
            if len(values) >= 2:
                return values
    raise ValueError(f"missing candidate records for {row_identity(row)}")


def tokenize_candidates(
    processor: Any, candidates: Sequence[Mapping[str, Any]]
) -> Dict[str, List[int]]:
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenized: Dict[str, List[int]] = {}
    for candidate in candidates:
        key = str(candidate["key"])
        text = str(candidate["text"])
        recorded = candidate.get("image_token_ids")
        prior_recorded = candidate.get("prior_token_ids")
        if isinstance(recorded, list) and recorded:
            values = [int(token) for token in recorded]
            if isinstance(prior_recorded, list) and values != [
                int(token) for token in prior_recorded
            ]:
                raise ValueError(f"image/prior candidate tokenization differs for {key}")
        else:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            values = [int(token) for token in ids]
        tokenized[key] = values
    return tokenized


def evaluate_row(
    spec: ModelSpec,
    model: Any,
    processor: Any,
    family: str,
    row: Mapping[str, Any],
    *,
    beta: float,
) -> Dict[str, Any]:
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    prompt = scoring_prompt(row)
    multimodal = candidate_vlm_inputs(model, processor, family, prompt, image)
    text_only = candidate_vlm_inputs(model, processor, family, prompt, None)
    candidates = candidate_specs(row)
    tokenized = tokenize_candidates(processor, candidates)
    started = time.time()
    choice = choose_candidate_with_nolan(
        model,
        multimodal,
        text_only,
        tokenized,
        beta=beta,
    )
    task = str(row["task"])
    gt = str(row["gt"])
    normalized_gt = gt.lower() if task == "qa" else gt.upper()
    saved_pred, saved_key, saved_correct = source_native_answer(row)
    candidate_native_key = (
        choice.native_key.lower() if task == "qa" else choice.native_key.upper()
    )
    native_key = saved_key if saved_key is not None else candidate_native_key
    native_correct = saved_correct if saved_key is not None else (
        candidate_native_key == normalized_gt
    )
    pred_key = choice.key.lower() if task == "qa" else choice.key.upper()
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
        "candidate_keys": list(tokenized),
        "candidate_token_ids": tokenized,
        "saved_native_pred": saved_pred,
        "saved_native_key": saved_key,
        "saved_native_correct": saved_correct,
        "native_key": native_key,
        "native_correct": native_correct,
        "candidate_native_key": candidate_native_key,
        "candidate_native_correct": candidate_native_key == normalized_gt,
        "candidate_native_matches_saved": candidate_native_key == saved_key,
        "pred_key": pred_key,
        "correct": pred_key == normalized_gt,
        "latency_ms": int((time.time() - started) * 1000),
        "nolan": {
            **choice.diagnostics,
            "scores": choice.scores,
            "native_scores": choice.native_scores,
            "prior_scores": choice.prior_scores,
            "prior_key": choice.prior_key,
        },
    }


def evaluate_trace_replay_row(
    spec: ModelSpec,
    row: Mapping[str, Any],
    trace_row: Mapping[str, Any],
    *,
    beta: float,
) -> Dict[str, Any]:
    """Replay an exact one-token candidate decision from a full-vocabulary trace."""
    candidates = candidate_specs(row)
    tokenized: Dict[str, List[int]] = {}
    for candidate in candidates:
        token_ids = candidate.get("image_token_ids")
        prior_token_ids = candidate.get("prior_token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != 1
            or not isinstance(prior_token_ids, list)
            or [int(value) for value in token_ids]
            != [int(value) for value in prior_token_ids]
        ):
            raise ValueError(
                "trace replay requires one-token candidates with identical "
                "image/prior tokenization"
            )
        tokenized[str(candidate["key"])] = [int(token_ids[0])]

    nolan = trace_row.get("nolan")
    if not isinstance(nolan, dict):
        raise ValueError(f"missing NoLan trace for {row_identity(row)}")
    trace_beta = float(nolan.get("beta", float("nan")))
    if abs(trace_beta - float(beta)) > 1e-9:
        raise ValueError(
            f"trace beta {trace_beta} does not match requested beta {float(beta)}"
        )
    steps = nolan.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"missing NoLan step trace for {row_identity(row)}")
    first_step: Optional[Mapping[str, Any]] = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and int(step.get("step", -1)) == 0
        ),
        None,
    )
    if first_step is None:
        raise ValueError(f"missing first NoLan step for {row_identity(row)}")
    alpha = float(first_step["alpha"])

    native_scores = {
        str(candidate["key"]): float(candidate["logp_image"])
        for candidate in candidates
    }
    prior_scores = {
        str(candidate["key"]): float(candidate["logp_prior"])
        for candidate in candidates
    }
    scores = {
        key: (1.0 + alpha) * native_scores[key] - alpha * prior_scores[key]
        for key in native_scores
    }
    candidate_native_key = max(native_scores, key=native_scores.get)
    prior_key = max(prior_scores, key=prior_scores.get)
    pred_key = max(scores, key=scores.get)
    task = str(row["task"])
    gt = str(row["gt"])
    normalized_gt = gt.lower() if task == "qa" else gt.upper()
    saved_pred, saved_key, saved_correct = source_native_answer(row)
    candidate_native_key = (
        candidate_native_key.lower()
        if task == "qa"
        else candidate_native_key.upper()
    )
    native_key = saved_key if saved_key is not None else candidate_native_key
    native_correct = saved_correct if saved_key is not None else (
        candidate_native_key == normalized_gt
    )
    prior_key = prior_key.lower() if task == "qa" else prior_key.upper()
    pred_key = pred_key.lower() if task == "qa" else pred_key.upper()
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
        "prompt": scoring_prompt(row),
        "candidate_keys": list(tokenized),
        "candidate_token_ids": tokenized,
        "saved_native_pred": saved_pred,
        "saved_native_key": saved_key,
        "saved_native_correct": saved_correct,
        "native_key": native_key,
        "native_correct": native_correct,
        "candidate_native_key": candidate_native_key,
        "candidate_native_correct": candidate_native_key == normalized_gt,
        "candidate_native_matches_saved": candidate_native_key == saved_key,
        "pred_key": pred_key,
        "correct": pred_key == normalized_gt,
        "latency_ms": 0,
        "nolan": {
            "method": "official_tokenwise_constrained_candidate_trace_replay",
            "official_repository": OFFICIAL_NOLAN_REPOSITORY,
            "official_commit": OFFICIAL_NOLAN_COMMIT,
            "beta": float(beta),
            "distribution": "full_vocabulary",
            "candidate_constraint": list(tokenized),
            "full_vocabulary_alpha": alpha,
            "full_vocabulary_symmetric_kl": float(first_step["symmetric_kl"]),
            "scores": scores,
            "native_scores": native_scores,
            "prior_scores": prior_scores,
            "prior_key": prior_key,
        },
    }


def summarize(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status") or "") == "ok":
            groups[(str(row["task"]), str(row["side"]))].append(row)
    payload: Dict[str, Any] = {}
    for (task, side), values in sorted(groups.items()):
        native = sum(bool(row["native_correct"]) for row in values) / len(values)
        candidate_native = (
            sum(bool(row["candidate_native_correct"]) for row in values) / len(values)
        )
        nolan = sum(bool(row["correct"]) for row in values) / len(values)
        payload[f"{task}/{side}"] = {
            "n": len(values),
            "native_accuracy": native,
            "candidate_native_accuracy": candidate_native,
            "nolan_accuracy": nolan,
            "delta_accuracy_points": 100.0 * (nolan - native),
            "delta_from_candidate_native_points": 100.0
            * (nolan - candidate_native),
            "changed": sum(row["native_key"] != row["pred_key"] for row in values),
            "repairs": sum(
                not bool(row["native_correct"]) and bool(row["correct"])
                for row in values
            ),
            "harms": sum(
                bool(row["native_correct"]) and not bool(row["correct"])
                for row in values
            ),
            "candidate_native_changed": sum(
                row["candidate_native_key"] != row["pred_key"] for row in values
            ),
            "candidate_native_repairs": sum(
                not bool(row["candidate_native_correct"]) and bool(row["correct"])
                for row in values
            ),
            "candidate_native_harms": sum(
                bool(row["candidate_native_correct"]) and not bool(row["correct"])
                for row in values
            ),
            "candidate_native_matches_saved": sum(
                bool(row["candidate_native_matches_saved"]) for row in values
            ),
        }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models", default="configs/qwen_32b_instruct_nolan_official.json"
    )
    parser.add_argument("--beta", type=float, default=OFFICIAL_NOLAN_BETA)
    parser.add_argument("--tasks", nargs="+", choices=("mc", "qa"), default=("mc", "qa"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--token-trace",
        help=(
            "Optional official token-decoding JSONL. One-token candidates can "
            "reuse its full-vocabulary first-step alpha without loading the model."
        ),
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
        trace_by_identity: Optional[Dict[Tuple[str, str, str], Mapping[str, Any]]] = None
        if args.token_trace:
            trace_by_identity = {
                row_identity(row): row for row in load_jsonl(Path(args.token_trace))
            }
            model = processor = family = None
        else:
            model, processor, family = local_candidate_bundle(spec)
        mode = "w" if args.no_resume else "a"
        with output.open(mode, encoding="utf-8") as handle:
            for index, row in enumerate(selected, start=1):
                identity = row_identity(row)
                if identity in done:
                    continue
                try:
                    if trace_by_identity is not None:
                        trace_row = trace_by_identity.get(identity)
                        if trace_row is None:
                            raise ValueError(f"missing token trace for {identity}")
                        result = evaluate_trace_replay_row(
                            spec, row, trace_row, beta=args.beta
                        )
                    else:
                        result = evaluate_row(
                            spec,
                            model,
                            processor,
                            family,
                            row,
                            beta=args.beta,
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
                    f"[{index}/{len(selected)}] {identity}: "
                    f"{result['status']} "
                    f"{result.get('native_key')}->{result.get('pred_key')}",
                    flush=True,
                )
        results = load_jsonl(output)

    summary = {
        "input": str(Path(args.input)),
        "output": str(output),
        "model": spec.name,
        "beta": float(args.beta),
        "protocol": "official_full_vocab_tokenwise_constrained_candidates",
        "token_trace": str(Path(args.token_trace)) if args.token_trace else None,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "metrics": summarize(results),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
