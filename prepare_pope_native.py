import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


PROTOCOLS = ("random", "popular", "adversarial")


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def download_image(filename: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_image(destination)
        return
    url = f"https://images.cocodataset.org/val2014/{filename}"
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
                "--connect-timeout",
                "20",
                "--max-time",
                "180",
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


def load_protocol(repo: Path, protocol: str) -> List[Dict[str, Any]]:
    path = repo / "output/coco" / f"coco_pope_{protocol}.json"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        label = str(row.get("label", "")).lower()
        if label not in {"yes", "no"}:
            raise ValueError(f"invalid {protocol} label: {label!r}")
    return rows


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    repo = Path(args.repo)
    output_root = Path(args.output_root)
    images_root = output_root / "images"
    records = []
    protocol_counts = {}
    for protocol in PROTOCOLS:
        rows = load_protocol(repo, protocol)
        protocol_counts[protocol] = len(rows)
        for row in rows:
            filename = str(row["image"])
            records.append(
                {
                    "protocol": protocol,
                    "question_id": int(row["question_id"]),
                    "image": filename,
                    "image_path": str(images_root / filename),
                    "question": str(row["text"]),
                    "label": str(row["label"]).lower(),
                }
            )

    unique_images = sorted({record["image"] for record in records})
    if args.download_images:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(download_image, filename, images_root / filename): filename
                for filename in unique_images
            }
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                if index % 50 == 0:
                    print(f"downloaded {index}/{len(unique_images)} images", flush=True)

    canonical = json.dumps(
        [
            [
                record["protocol"],
                record["question_id"],
                record["image"],
                record["question"],
                record["label"],
            ]
            for record in records
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    manifest = {
        "benchmark": "POPE",
        "official_repo": "https://github.com/RUCAIBox/POPE",
        "protocols": list(PROTOCOLS),
        "protocol_counts": protocol_counts,
        "question_count": len(records),
        "unique_image_count": len(unique_images),
        "selection": "All released COCO questions in random, popular, and adversarial protocols.",
        "response_blind_selection": True,
        "selection_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "dataset": str(output_root / "pope_native.jsonl"),
        "images_root": str(images_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    Path(manifest["dataset"]).write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the full native POPE protocols.")
    parser.add_argument("--repo", default="downloads/hallucination_refs/POPE")
    parser.add_argument("--output-root", default="data/pope_native")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
