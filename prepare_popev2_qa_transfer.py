import argparse
import io
import json
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


DEFAULT_REPO = Path("downloads/hallucination_refs/POPE")


def load_pairs(annotation_path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    grouped: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        entry = grouped.setdefault(image_id, {"image_id": image_id})
        label = str(row["label"]).strip().lower()
        if label not in {"yes", "no"}:
            raise ValueError(f"unsupported label for image {image_id}: {label}")
        entry[label] = row

    pairs = []
    for image_id, entry in grouped.items():
        if "yes" not in entry or "no" not in entry:
            raise ValueError(f"image {image_id} does not have both POPEv2 sides")
        if entry["yes"]["query"] != entry["no"]["query"]:
            raise ValueError(f"image {image_id} has inconsistent paired questions")
        pairs.append(entry)
    return sorted(pairs, key=lambda value: value["image_id"])


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def download_coco_image(image_id: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_image(destination)
        return
    filename = f"{image_id:012d}.jpg"
    url = f"https://images.cocodataset.org/train2017/{filename}"
    with tempfile.NamedTemporaryFile(
        suffix=".jpg", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(
            [
                "curl",
                "-kfsSL",
                "--http1.1",
                "--retry",
                "5",
                "--retry-all-errors",
                "-o",
                str(temporary),
                url,
            ],
            check=True,
        )
        validate_image(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_as_png(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="PNG")


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    repo = Path(args.repo)
    annotation_path = repo / "POPEv2/dataset/annotations.json"
    image_zip_path = repo / "POPEv2/dataset/images.zip"
    output_root = Path(args.output_root)
    cache_root = output_root / "cache"
    extracted_root = cache_root / "modified"
    original_root = cache_root / "coco_train2017"
    dataset_path = output_root / "popev2_qa.jsonl"
    images_root = output_root / "images"

    pairs = load_pairs(annotation_path)
    rng = random.Random(int(args.seed))
    rng.shuffle(pairs)
    if int(args.limit) > 0:
        pairs = pairs[: int(args.limit)]

    extracted_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(image_zip_path) as archive:
        for pair in pairs:
            filename = f"{int(pair['image_id']):012d}.jpg"
            member = f"images/{filename}"
            destination = extracted_root / filename
            if not destination.exists():
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
            validate_image(destination)

    records: List[Dict[str, Any]] = []
    for rank, pair in enumerate(pairs, start=1):
        image_id = int(pair["image_id"])
        filename = f"{image_id:012d}.jpg"
        original = original_root / filename
        download_coco_image(image_id, original)

        pair_id = f"Pair {rank:04d}"
        pair_dir = images_root / "POPEv2" / pair_id.replace(" ", "_")
        save_as_png(extracted_root / filename, pair_dir / "counterfactual.png")
        save_as_png(original, pair_dir / "commonsense.png")

        question = str(pair["no"]["query"])
        target_object = str(pair["no"]["target_object"])
        records.append(
            {
                "pair_id": pair_id,
                "pair_name": f"POPEv2 removed/present: {target_object}",
                "category": "Object Hallucination",
                "subcategory": "POPEv2",
                "direct_qa": {
                    "question": question,
                    "counterfactual_gt": "no",
                    "commonsense_gt": "yes",
                },
                "source": {
                    "benchmark": "POPEv2",
                    "official_repo": "https://github.com/RUCAIBox/POPE",
                    "image_id": image_id,
                    "target_object": target_object,
                },
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "benchmark": "POPEv2",
        "official_repo": "https://github.com/RUCAIBox/POPE",
        "source_annotation": str(annotation_path),
        "source_modified_images": str(image_zip_path),
        "coco_original_url": "https://images.cocodataset.org/train2017/{image_id:012d}.jpg",
        "seed": int(args.seed),
        "pair_count": len(records),
        "dataset": str(dataset_path),
        "images_root": str(images_root),
        "inference_fairness": (
            "Each inference receives only its current image and the native POPEv2 yes/no question. "
            "Pair identity and side are used only for grouped evaluation."
        ),
        "selected_image_ids": [int(pair["image_id"]) for pair in pairs],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a paired POPEv2 QA transfer set.")
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--output-root", default="data/popev2_qa_transfer")
    parser.add_argument("--limit", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
