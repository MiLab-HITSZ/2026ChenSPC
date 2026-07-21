#!/usr/bin/env python3
import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

from cdh_caption_claims import split_multiple_choice_options
from evaluate_cdh_bench import (
    _build_candidate_scoring_prompt,
    _candidate_logprobs_shared_prefix,
    _load_model_specs,
    _local_candidate_bundle,
    _make_mfcd_high_frequency_image,
    _make_vcd_degraded_image,
)


VERSION = "candidate_view_scores_v2"
LOW_VIEWS = {"blur_downsample", "blur", "downsample", "low_contrast"}
HIGH_VIEWS = {"high_pass", "edges", "sharpen"}
NEUTRAL_VIEWS = {"white", "black", "development_mean"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return str(row.get("task")), str(row.get("pair_id")), str(row.get("side"))


def normalize_item_options(item: Dict[str, Any]) -> Dict[str, Any]:
    output = json.loads(json.dumps(item))
    mc = output.get("multiple_choice") or {}
    if isinstance(mc.get("options"), list):
        mc["options"] = split_multiple_choice_options(mc["options"])
    return output


def view_image(
    image: Image.Image,
    mode: str,
    mean_image: Image.Image | None = None,
) -> Image.Image:
    if mode in LOW_VIEWS:
        return _make_vcd_degraded_image(image, mode)
    if mode in HIGH_VIEWS:
        return _make_mfcd_high_frequency_image(image, mode)
    if mode == "white":
        return Image.new("RGB", image.size, (255, 255, 255))
    if mode == "black":
        return Image.new("RGB", image.size, (0, 0, 0))
    if mode == "development_mean":
        if mean_image is None:
            raise ValueError("development_mean requires --mean-image")
        return mean_image.resize(image.size, Image.Resampling.LANCZOS).copy()
    raise ValueError(f"unknown view mode: {mode}")


def existing_keys(path: Path) -> set[Tuple[str, str, str]]:
    if not path.exists():
        return set()
    return {
        row_key(row)
        for row in read_jsonl(path)
        if row.get("status") == "ok"
    }


def select_rows(
    rows: Sequence[Dict[str, Any]],
    tasks: set[str],
    shard_count: int,
    shard_index: int,
) -> List[Dict[str, Any]]:
    eligible = sorted(
        (
            row
            for row in rows
            if row.get("status") == "ok"
            and str(row.get("task")) in tasks
            and isinstance(row.get("cp_vbc"), dict)
        ),
        key=row_key,
    )
    return [
        row
        for index, row in enumerate(eligible)
        if index % shard_count == shard_index
    ]


def collect_row(
    row: Dict[str, Any],
    item: Dict[str, Any],
    model: Any,
    processor: Any,
    family: str,
    modes: Sequence[str],
    mean_image: Image.Image | None = None,
) -> Dict[str, Any]:
    image = Image.open(str(row["image_path"])).convert("RGB")
    if family == "qwen3_vl":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    views = {mode: view_image(image, mode, mean_image) for mode in modes}
    prompt = _build_candidate_scoring_prompt(str(row["task"]), item)
    source = row["cp_vbc"]
    source_candidates = source.get("candidates") or []
    if not source_candidates:
        raise ValueError("base CPRC row has no candidates")
    scores_by_mode = {
        mode: _candidate_logprobs_shared_prefix(
            model,
            processor,
            prompt,
            transformed,
            [str(candidate["text"]) for candidate in source_candidates],
            family=family,
        )
        for mode, transformed in views.items()
    }
    candidates = []
    for candidate_index, candidate in enumerate(source_candidates):
        candidate_views = {}
        for mode in views:
            score = scores_by_mode[mode][candidate_index]
            candidate_views[mode] = {
                "logprob": score["logprob"],
                "avg_logprob": score["avg_logprob"],
                "token_count": score["token_count"],
                "token_ids": score["token_ids"],
            }
        candidates.append(
            {
                "key": candidate["key"],
                "text": candidate["text"],
                "logp_image": candidate["logp_image"],
                "avg_logp_image": candidate.get("avg_logp_image"),
                "logp_prior": candidate.get("logp_prior"),
                "avg_logp_prior": candidate.get("avg_logp_prior"),
                "views": candidate_views,
            }
        )
    return {
        "version": VERSION,
        "status": "ok",
        "model_name": row.get("model_name"),
        "model": row.get("model"),
        "family": family,
        "pair_id": row["pair_id"],
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "task": row["task"],
        "side": row["side"],
        "image_path": row["image_path"],
        "question": row.get("question"),
        "gt": row["gt"],
        "view_collection": {
            "score_source": "frozen_cprc_image_scores_plus_same_image_derived_views",
            "base_run": row.get("run"),
            "baseline_key": source.get("baseline_key"),
            "baseline_pred": source.get("baseline_pred"),
            "scoring_prompt": prompt,
            "view_modes": list(modes),
            "candidates": candidates,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--dataset", default="CDH-Bench.revised.strict.jsonl")
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--view", action="append", required=True)
    parser.add_argument("--mean-image", default="")
    parser.add_argument("--tasks", default="qa,mc")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-file", default="")
    args = parser.parse_args()

    modes = list(dict.fromkeys(str(mode).strip() for mode in args.view))
    unknown = set(modes) - LOW_VIEWS - HIGH_VIEWS - NEUTRAL_VIEWS
    if unknown:
        raise SystemExit(f"unknown view modes: {sorted(unknown)}")
    shard_count = max(1, int(args.shard_count))
    shard_index = int(args.shard_index)
    if not 0 <= shard_index < shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")
    mean_image = None
    if "development_mean" in modes:
        if not args.mean_image:
            raise SystemExit("--mean-image is required for development_mean")
        mean_image = Image.open(args.mean_image).convert("RGB")

    specs = _load_model_specs(args.models)
    if len(specs) != 1 or specs[0].backend != "transformers":
        raise SystemExit("exactly one local transformers model is required")
    spec = specs[0]
    dataset_path = Path(args.dataset)
    items = {
        str(item["pair_id"]): normalize_item_options(item)
        for item in read_jsonl(dataset_path)
    }
    base_path = Path(args.base_results)
    tasks = {value.strip() for value in str(args.tasks).split(",") if value.strip()}
    selected = select_rows(
        read_jsonl(base_path), tasks, shard_count, shard_index
    )
    output = Path(args.output)
    completed = existing_keys(output)
    pending = [row for row in selected if row_key(row) not in completed]

    model, processor, family = _local_candidate_bundle(spec)
    errors = 0
    for index, row in enumerate(tqdm(pending, desc="Candidate view scores"), start=1):
        try:
            record = collect_row(
                row,
                items[str(row["pair_id"])],
                model,
                processor,
                family,
                modes,
                mean_image,
            )
        except Exception as exc:
            errors += 1
            record = {
                "version": VERSION,
                "status": "error",
                "pair_id": row.get("pair_id"),
                "task": row.get("task"),
                "side": row.get("side"),
                "gt": row.get("gt"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        append_jsonl(output, record)
        if args.progress_file:
            progress = {
                "version": VERSION,
                "output": str(output),
                "base_results": str(base_path),
                "base_results_sha256": sha256(base_path),
                "dataset": str(dataset_path),
                "dataset_sha256": sha256(dataset_path),
                "model": spec.name,
                "family": family,
                "views": modes,
                "shard_count": shard_count,
                "shard_index": shard_index,
                "selected": len(selected),
                "preexisting": len(completed),
                "completed_this_run": index,
                "errors": errors,
            }
            progress_path = Path(args.progress_file)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    if errors:
        raise SystemExit(f"candidate view collection completed with {errors} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
