import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image


DEFAULT_REPO = Path("downloads/hallucination_refs/HallusionBench")


def answer_key(value: Any) -> str:
    key = str(value).strip()
    if key == "1":
        return "yes"
    if key == "0":
        return "no"
    raise ValueError(f"unsupported HallusionBench answer: {value!r}")


def paired_questions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, List[Dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        key = (
            str(row["category"]),
            str(row["subcategory"]),
            str(row["set_id"]),
            str(row["question_id"]),
            str(row["question"]),
        )
        grouped[key][str(row["visual_input"])].append(row)

    pairs: List[Dict[str, Any]] = []
    for key, variants in grouped.items():
        easy_rows = sorted(
            variants.get("1", []), key=lambda row: (str(row["figure_id"]), str(row["filename"]))
        )
        hard_rows = sorted(
            variants.get("2", []), key=lambda row: (str(row["figure_id"]), str(row["filename"]))
        )
        if not easy_rows or not hard_rows:
            continue
        easy = easy_rows[0]
        hard = next(
            (row for row in hard_rows if answer_key(row["gt_answer"]) != answer_key(easy["gt_answer"])),
            None,
        )
        if hard is None or not easy.get("filename") or not hard.get("filename"):
            continue
        pairs.append({"key": key, "easy": easy, "hard": hard})
    return pairs


def stratified_sample(
    pairs: List[Dict[str, Any]], limit: int, seed: int
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        easy = pair["easy"]
        buckets[(str(easy["category"]), str(easy["subcategory"]))].append(pair)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)

    selected: List[Dict[str, Any]] = []
    active = sorted(buckets)
    while active and (limit <= 0 or len(selected) < limit):
        next_active = []
        for bucket in active:
            if buckets[bucket] and (limit <= 0 or len(selected) < limit):
                selected.append(buckets[bucket].pop())
            if buckets[bucket]:
                next_active.append(bucket)
        active = next_active
    return selected


def save_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        background.convert("RGB").save(destination, format="PNG")


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    repo = Path(args.repo)
    rows = json.loads((repo / "HallusionBench.json").read_text(encoding="utf-8"))
    candidates = paired_questions(rows)
    selected = stratified_sample(candidates, int(args.limit), int(args.seed))

    output_root = Path(args.output_root)
    images_root = output_root / "images"
    records: List[Dict[str, Any]] = []
    distribution: Counter = Counter()
    for rank, pair in enumerate(selected, start=1):
        easy = pair["easy"]
        hard = pair["hard"]
        source_category = str(easy["category"])
        source_subcategory = str(easy["subcategory"])
        subcategory = f"{source_category}-{source_subcategory}"
        distribution[subcategory] += 1
        pair_id = f"Pair {rank:04d}"
        pair_dir = images_root / subcategory / pair_id.replace(" ", "_")
        easy_path = repo / "hallusion_bench" / str(easy["filename"]).removeprefix("./")
        hard_path = repo / "hallusion_bench" / str(hard["filename"]).removeprefix("./")
        save_image(easy_path, pair_dir / "commonsense.png")
        save_image(hard_path, pair_dir / "counterfactual.png")

        cv_group = f"{source_category}/{source_subcategory}/set-{easy['set_id']}"
        records.append(
            {
                "pair_id": pair_id,
                "cv_group": cv_group,
                "pair_name": f"HallusionBench easy/hard: {cv_group}, q{easy['question_id']}",
                "category": f"HallusionBench {source_category}",
                "subcategory": subcategory,
                "direct_qa": {
                    "question": str(easy["question"]),
                    "counterfactual_gt": answer_key(hard["gt_answer"]),
                    "commonsense_gt": answer_key(easy["gt_answer"]),
                },
                "source": {
                    "benchmark": "HallusionBench",
                    "official_repo": "https://github.com/tianyi-lab/HallusionBench",
                    "category": source_category,
                    "subcategory": source_subcategory,
                    "set_id": str(easy["set_id"]),
                    "question_id": str(easy["question_id"]),
                    "easy_figure_id": str(easy["figure_id"]),
                    "hard_figure_id": str(hard["figure_id"]),
                },
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / "hallusionbench_qa.jsonl"
    dataset_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "benchmark": "HallusionBench",
        "official_repo": "https://github.com/tianyi-lab/HallusionBench",
        "selection": "Exact same question; visual_input=1 easy and visual_input=2 hard; opposite GT.",
        "seed": int(args.seed),
        "eligible_pair_count": len(candidates),
        "selected_pair_count": len(records),
        "selected_pair_keys": [list(pair["key"]) for pair in selected],
        "distribution": dict(sorted(distribution.items())),
        "dataset": str(dataset_path),
        "images_root": str(images_root),
        "inference_fairness": (
            "Each inference receives only its current image and the native HallusionBench yes/no "
            "question. Easy/hard pairing and set-level cv_group are evaluation-only metadata."
        ),
    }
    canonical = json.dumps(
        manifest["selected_pair_keys"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest["selection_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    manifest["response_blind_selection"] = True
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare paired HallusionBench QA transfer data.")
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--output-root", default="data/hallusionbench_qa_transfer")
    parser.add_argument("--limit", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
