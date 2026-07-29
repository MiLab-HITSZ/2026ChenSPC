#!/usr/bin/env python3
"""Evaluate official NoLan at the answer token on external MC/QA benchmarks.

Every benchmark uses an answer-only protocol. The matched beta=0 prediction
and the beta=0.8 prediction therefore come from the same full-vocabulary
next-token distributions, before any candidate filtering.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image

from spc_vlm_runtime import load_model_specs, local_candidate_bundle
from evaluate_official_nolan_qwen3vl import generated_answer_key, scoring_prompt
from official_nolan_qwen3vl import OFFICIAL_NOLAN_BETA, nolan_adjust_logits


Identity = Tuple[str, str, str]


def row_identity(row: Mapping[str, Any]) -> Identity:
    return (
        str(row.get("pair_id") or ""),
        str(row.get("task") or ""),
        str(row.get("side") or ""),
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_identities(path: Path) -> set[Identity]:
    if not path.exists():
        return set()
    return {
        row_identity(row)
        for row in load_jsonl(path)
        if str(row.get("status") or "") == "ok"
    }


def batches(values: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def qwen_batch_inputs(
    processor: Any,
    prompts: Sequence[str],
    images: Sequence[Image.Image] | None,
) -> Dict[str, Any]:
    conversations = []
    for index, prompt in enumerate(prompts):
        content: List[Dict[str, Any]] = []
        if images is not None:
            content.append({"type": "image", "image": images[index]})
        content.append({"type": "text", "text": prompt})
        conversations.append([{"role": "user", "content": content}])
    return processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )


def next_token_logits(model: Any, inputs: Mapping[str, Any]) -> Any:
    moved = {
        key: (value.to(model.device) if hasattr(value, "to") else value)
        for key, value in inputs.items()
    }
    outputs = model(
        **moved,
        use_cache=False,
        logits_to_keep=1,
        return_dict=True,
    )
    return outputs.logits[:, -1, :]


def parse_token(processor: Any, task: str, token_id: int) -> Tuple[str, str | None]:
    tokenizer = getattr(processor, "tokenizer", processor)
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return text, generated_answer_key(task, text)


def evaluate_batch(
    model: Any,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    beta: float,
) -> List[Dict[str, Any]]:
    import torch

    prompts = [scoring_prompt(row) for row in rows]
    images = []
    for row in rows:
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = Path.cwd() / path
        with Image.open(path) as image:
            images.append(
                image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            )

    started = time.time()
    with torch.inference_mode():
        image_logits = next_token_logits(
            model, qwen_batch_inputs(processor, prompts, images)
        )
        prior_logits = next_token_logits(
            model, qwen_batch_inputs(processor, prompts, None)
        )
        adjusted, gamma, alpha = nolan_adjust_logits(
            image_logits, prior_logits, beta=beta
        )
        native_ids = torch.argmax(image_logits, dim=-1).detach().cpu().tolist()
        prior_ids = torch.argmax(prior_logits, dim=-1).detach().cpu().tolist()
        nolan_ids = torch.argmax(adjusted, dim=-1).detach().cpu().tolist()
        gamma_values = gamma.detach().cpu().tolist()
        alpha_values = alpha.detach().cpu().tolist()
    latency_ms = int((time.time() - started) * 1000)

    results: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        task = str(row["task"])
        gt = str(row["gt"]).lower() if task == "qa" else str(row["gt"]).upper()
        native_text, native_key = parse_token(processor, task, native_ids[index])
        prior_text, prior_key = parse_token(processor, task, prior_ids[index])
        pred_text, pred_key = parse_token(processor, task, nolan_ids[index])
        results.append(
            {
                "benchmark": row.get("benchmark"),
                "source_pair_id": row.get("source_pair_id"),
                "pair_id": row.get("pair_id"),
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "protocol": row.get("protocol"),
                "task": task,
                "side": row.get("side"),
                "image_path": row.get("image_path"),
                "question": row.get("question"),
                "gt": str(row["gt"]),
                "status": "ok",
                "prompt": prompts[index],
                "native_pred": native_text,
                "native_key": native_key,
                "native_correct": native_key == gt,
                "prior_pred": prior_text,
                "prior_key": prior_key,
                "pred": pred_text,
                "pred_key": pred_key,
                "correct": pred_key == gt,
                "latency_ms_per_batch": latency_ms,
                "nolan": {
                    "method": "official_token_level_nolan",
                    "beta": float(beta),
                    "distribution": "full_vocabulary",
                    "prior_condition": "text_only",
                    "answer_protocol": "single_token",
                    "native_token_id": int(native_ids[index]),
                    "prior_token_id": int(prior_ids[index]),
                    "nolan_token_id": int(nolan_ids[index]),
                    "symmetric_kl": float(gamma_values[index]),
                    "alpha": float(alpha_values[index]),
                },
            }
        )
    return results


def binary_f1(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    tp = fp = fn = 0
    for row in rows:
        prediction = row.get(key)
        truth = str(row.get("gt") or "").lower()
        tp += int(prediction == "yes" and truth == "yes")
        fp += int(prediction == "yes" and truth == "no")
        fn += int(prediction != "yes" and truth == "yes")
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def group_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    native_accuracy = sum(bool(row["native_correct"]) for row in rows) / len(rows)
    accuracy = sum(bool(row["correct"]) for row in rows) / len(rows)
    return {
        "n": len(rows),
        "native_accuracy": native_accuracy,
        "nolan_accuracy": accuracy,
        "delta_accuracy_points": 100.0 * (accuracy - native_accuracy),
        "changed": sum(row.get("native_key") != row.get("pred_key") for row in rows),
        "repairs": sum(
            not bool(row["native_correct"]) and bool(row["correct"]) for row in rows
        ),
        "harms": sum(
            bool(row["native_correct"]) and not bool(row["correct"]) for row in rows
        ),
        "invalid_native": sum(row.get("native_key") is None for row in rows),
        "invalid_nolan": sum(row.get("pred_key") is None for row in rows),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_benchmark: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status") or "") == "ok":
            by_benchmark[str(row.get("benchmark") or "")].append(row)
    payload: Dict[str, Any] = {}
    for benchmark, values in sorted(by_benchmark.items()):
        entry: Dict[str, Any] = {"overall": group_metrics(values)}
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            groups[f"task:{row['task']}"].append(row)
            groups[f"side:{row['side']}"].append(row)
            if row.get("protocol"):
                groups[f"protocol:{row['protocol']}"].append(row)
        entry["groups"] = {
            name: group_metrics(group) for name, group in sorted(groups.items())
        }
        if benchmark == "POPE":
            entry["native_f1"] = binary_f1(values, "native_key")
            entry["nolan_f1"] = binary_f1(values, "pred_key")
            entry["delta_f1_points"] = 100.0 * (
                entry["nolan_f1"] - entry["native_f1"]
            )
        payload[benchmark] = entry
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--models", default="configs/qwen_32b_instruct_nolan_official.json"
    )
    parser.add_argument("--beta", type=float, default=OFFICIAL_NOLAN_BETA)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = [
        row for row in load_jsonl(Path(args.input)) if row.get("status") == "ok"
    ]
    source.sort(key=row_identity)
    if args.limit > 0:
        source = source[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.no_resume else completed_identities(output)

    if not args.summary_only:
        specs = load_model_specs(args.models)
        if len(specs) != 1:
            raise ValueError("the evaluator requires exactly one model")
        model, processor, family = local_candidate_bundle(specs[0])
        if family != "qwen3_vl":
            raise ValueError("the batched external evaluator currently supports Qwen3-VL")
        processor.tokenizer.padding_side = "left"
        pending = [row for row in source if row_identity(row) not in done]
        mode = "w" if args.no_resume else "a"
        with output.open(mode, encoding="utf-8") as handle:
            completed = len(source) - len(pending)
            for batch in batches(pending, int(args.batch_size)):
                results = evaluate_batch(model, processor, batch, float(args.beta))
                for result in results:
                    handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                handle.flush()
                completed += len(results)
                print(f"[{completed}/{len(source)}] completed", flush=True)

    results = load_jsonl(output)
    summary = {
        "input": str(Path(args.input)),
        "output": str(output),
        "beta": float(args.beta),
        "answer_protocol": "single_token_full_vocabulary",
        "metrics": summarize(results),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
