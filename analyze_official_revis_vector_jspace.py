#!/usr/bin/env python3
"""Project an official REVIS vector through a checkpoint's final unembedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _token_record(tokenizer: Any, token_id: int, score: float) -> dict[str, Any]:
    return {
        "token_id": int(token_id),
        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
        "decoded": tokenizer.decode([int(token_id)]),
        "score": float(score),
    }


def _target_records(tokenizer: Any, scores: Any, texts: list[str]) -> list[dict[str, Any]]:
    records = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for text in texts:
        token_ids = tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))
        key = (text, token_ids)
        if not token_ids or key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "text": text,
                "token_ids": list(token_ids),
                "tokens": [tokenizer.convert_ids_to_tokens(value) for value in token_ids],
                "scores": [float(scores[value]) for value in token_ids],
                "mean_score": float(sum(float(scores[value]) for value in token_ids) / len(token_ids)),
            }
        )
    return records


def analyze(
    model_path: Path,
    vector_file: Path,
    vector_index: int,
    top_k: int,
    scan_all_indices: bool = False,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    lm_head_name = "lm_head.weight"
    norm_name = "model.language_model.norm.weight"
    if weight_map[lm_head_name] != weight_map[norm_name]:
        raise RuntimeError("lm_head and final norm are expected in the same shard")

    vectors = torch.load(vector_file, map_location="cpu", weights_only=False)
    visual = vectors[int(vector_index)]["visual"].float()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config") or config
    eps = float(text_config.get("rms_norm_eps", 1e-6))

    shard = model_path / weight_map[lm_head_name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor(norm_name).float()
        lm_head = handle.get_tensor(lm_head_name).float()

    if visual.numel() != norm_weight.numel() or lm_head.shape[1] != visual.numel():
        raise ValueError(
            f"dimension mismatch: vector={visual.numel()}, norm={norm_weight.numel()}, "
            f"lm_head={tuple(lm_head.shape)}"
        )
    normalized = visual * torch.rsqrt(visual.pow(2).mean() + eps) * norm_weight
    scores = torch.mv(lm_head, normalized)
    positive_scores, positive_ids = torch.topk(scores, k=top_k)
    negative_scores, negative_ids = torch.topk(-scores, k=top_k)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    targets = _target_records(
        tokenizer,
        scores,
        ["yes", " yes", "Yes", " Yes", "no", " no", "No", " No"],
    )
    target_scores = {record["text"]: record["mean_score"] for record in targets}
    contrasts = []
    for yes_text, no_text in (("yes", "no"), (" yes", " no"), ("Yes", "No"), (" Yes", " No")):
        if yes_text in target_scores and no_text in target_scores:
            contrasts.append(
                {
                    "yes_text": yes_text,
                    "no_text": no_text,
                    "yes_minus_no": float(target_scores[yes_text] - target_scores[no_text]),
                }
            )
    all_index_contrasts = []
    if scan_all_indices:
        target_texts = ["yes", " yes", "Yes", " Yes", "no", " no", "No", " No"]
        target_ids = {
            int(token_id)
            for text in target_texts
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        }
        target_rows = {token_id: lm_head[token_id] for token_id in target_ids}
        for index_value in sorted(int(value) for value in vectors):
            candidate = vectors[index_value]["visual"].float()
            candidate_normed = candidate * torch.rsqrt(candidate.pow(2).mean() + eps) * norm_weight
            candidate_scores = {
                token_id: float(torch.dot(row, candidate_normed))
                for token_id, row in target_rows.items()
            }
            candidate_targets = _target_records(tokenizer, candidate_scores, target_texts)
            score_by_text = {record["text"]: record["mean_score"] for record in candidate_targets}
            candidate_contrasts = {
                f"{yes_text!r}-{no_text!r}": float(score_by_text[yes_text] - score_by_text[no_text])
                for yes_text, no_text in (("yes", "no"), (" yes", " no"), ("Yes", "No"), (" Yes", " No"))
            }
            all_index_contrasts.append(
                {
                    "vector_index": index_value,
                    "yes_minus_no": candidate_contrasts,
                    "mean_yes_minus_no": float(
                        sum(candidate_contrasts.values()) / len(candidate_contrasts)
                    ),
                }
            )
    return {
        "diagnostic": "static_final_rmsnorm_unembedding_projection",
        "warning": (
            "This is a static logit-lens projection of a mid-layer intervention vector. "
            "It does not model transport through later decoder blocks and is not used for inference."
        ),
        "model_path": str(model_path),
        "vector_file": str(vector_file),
        "vector_index": int(vector_index),
        "hidden_width": int(visual.numel()),
        "vector_norm": float(torch.linalg.vector_norm(visual)),
        "normalized_vector_norm": float(torch.linalg.vector_norm(normalized)),
        "rms_norm_eps": eps,
        "top_promoted": [
            _token_record(tokenizer, int(token_id), float(score))
            for token_id, score in zip(positive_ids, positive_scores)
        ],
        "top_suppressed": [
            _token_record(tokenizer, int(token_id), -float(score))
            for token_id, score in zip(negative_ids, negative_scores)
        ],
        "yes_no_tokens": targets,
        "yes_no_contrasts": contrasts,
        "all_vector_yes_no_contrasts": all_index_contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vector-file", type=Path, required=True)
    parser.add_argument("--vector-index", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--scan-all-indices", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = analyze(
        args.model_path,
        args.vector_file,
        args.vector_index,
        args.top_k,
        scan_all_indices=args.scan_all_indices,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
