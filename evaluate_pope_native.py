import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from cprc_vlm_runtime import (
    build_candidate_prompt,
    candidate_key,
    candidate_vlm_inputs,
    generate_qa,
    load_model_specs,
    local_candidate_bundle,
)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed(path: Path) -> set:
    if not path.exists():
        return set()
    return {
        (str(row["protocol"]), int(row["question_id"]))
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
        if row.get("status") == "ok"
    }


def score_qa_candidates(
    spec: Any,
    item: Dict[str, Any],
    image_path: str,
    image_preprocessing: str = "fixed_512",
) -> tuple:
    import torch
    from PIL import Image

    model, processor, family = local_candidate_bundle(spec)
    if family != "qwen3_vl":
        raise ValueError("the native POPE fast path currently requires Qwen3-VL")
    prompt = build_candidate_prompt(str(item["direct_qa"]["question"]))
    image = Image.open(image_path).convert("RGB")
    if image_preprocessing == "fixed_512":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    elif image_preprocessing != "model_native":
        raise ValueError(f"unknown image preprocessing: {image_preprocessing}")
    started = time.time()

    token_ids = {}
    for key in ("yes", "no"):
        ids = processor.tokenizer.encode(f" {key}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"expected one token for {key!r}, got {ids}")
        token_ids[key] = int(ids[0])

    distributions = {}
    for source, current_image in (("image", image), ("prior", None)):
        inputs = candidate_vlm_inputs(
            model, processor, family, prompt, current_image
        )
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, return_dict=True)
            logp = torch.log_softmax(output.logits[0, -1, :].float(), dim=-1)
        distributions[source] = {
            key: float(logp[token_id].item()) for key, token_id in token_ids.items()
        }

    candidates = []
    for key in ("yes", "no"):
        candidates.append(
            {
                "key": key,
                "text": f" {key}",
                "logp_image": distributions["image"][key],
                "avg_logp_image": distributions["image"][key],
                "logp_prior": distributions["prior"][key],
                "avg_logp_prior": distributions["prior"][key],
                "image_token_count": 1,
                "prior_token_count": 1,
                "image_token_ids": [token_ids[key]],
                "prior_token_ids": [token_ids[key]],
            }
        )
    return candidates, int((time.time() - started) * 1000)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full native POPE with candidate scores.")
    parser.add_argument("--dataset", default="data/pope_native/pope_native.jsonl")
    parser.add_argument("--models", default="configs/qwen_32b_instruct_bf16_fair.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")
    specs = load_model_specs(args.models)
    if len(specs) != 1 or specs[0].backend != "transformers":
        raise SystemExit("exactly one transformers model is required per shard")
    spec = specs[0]
    rows: List[Dict[str, Any]] = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
    ]
    selected = [
        row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    output = Path(args.output)
    completed = load_completed(output)

    for position, source in enumerate(selected, start=1):
        key = (str(source["protocol"]), int(source["question_id"]))
        if key in completed:
            continue
        item = {
            "direct_qa": {
                "question": str(source["question"]),
                "counterfactual_gt": str(source["label"]),
                "commonsense_gt": str(source["label"]),
            }
        }
        started = time.time()
        status, baseline_text, baseline_raw, baseline_ms = generate_qa(
            spec,
            question=str(source["question"]),
            image_path=str(source["image_path"]),
            retry=int(args.retry),
        )
        record: Dict[str, Any] = {
            "protocol": source["protocol"],
            "question_id": int(source["question_id"]),
            "pair_id": f"{source['protocol']}-{int(source['question_id']):04d}",
            "category": "Object Hallucination",
            "subcategory": f"POPE-{source['protocol']}",
            "task": "qa",
            "side": "counterfactual",
            "image_path": source["image_path"],
            "question": source["question"],
            "gt": source["label"],
            "status": status,
            "model_name": spec.name,
            "model": spec.model,
            "pred": candidate_key(spec, "qa", baseline_text),
            "baseline_text": baseline_text,
            "baseline_raw": baseline_raw if status == "ok" else None,
            "baseline_latency_ms": baseline_ms,
        }
        if status == "ok":
            try:
                candidates, score_ms = score_qa_candidates(
                    spec, item, str(source["image_path"])
                )
                baseline_key = candidate_key(
                    spec, "qa", baseline_text
                )
                record["cp_vbc"] = {
                    "backend": "local_transformers_qwen3_vl_joint_next_token",
                    "model": spec.model,
                    "image_preprocessing": "fixed_512",
                    "mitigation": "score_collection_only",
                    "scoring_prompt": build_candidate_prompt(str(source["question"])),
                    "baseline_pred": baseline_text,
                    "baseline_key": baseline_key,
                    "image_top": max(candidates, key=lambda row: row["logp_image"])["key"],
                    "prior_top": max(candidates, key=lambda row: row["logp_prior"])["key"],
                    "candidates": candidates,
                }
                record["score_latency_ms"] = score_ms
            except Exception as error:
                record["status"] = "error"
                record["error"] = str(error)
        record["latency_ms"] = int((time.time() - started) * 1000)
        record["correct"] = bool(record["pred"] == record["gt"])
        append_jsonl(output, record)
        if position % 50 == 0:
            print(
                f"shard {args.shard_index}: {position}/{len(selected)}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
