#!/usr/bin/env python3
"""Download the COCO images referenced by the pinned official REVIS release."""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from PIL import Image


COCO_TRAIN2014 = "https://images.cocodataset.org/train2014"


def extraction_filenames(annotation_file: Path) -> list[str]:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    rows = payload["annotations"] if isinstance(payload, dict) else payload
    return sorted({f"{row['image_id']}.jpg" for row in rows})


def calibration_filenames(question_file: Path, max_samples: int) -> list[str]:
    names: set[str] = set()
    with question_file.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= max_samples:
                break
            row = json.loads(line)
            names.add(str(row["image"]))
    return sorted(names)


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_one(filename: str, destination: Path, retries: int) -> tuple[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    if valid_image(target):
        return filename, "cached"

    url = f"{COCO_TRAIN2014}/{filename}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "cdh-revis-reproduction/1.0"})
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                temporary.write_bytes(response.read())
            temporary.replace(target)
            if not valid_image(target):
                raise ValueError(f"downloaded file is not a valid image: {target}")
            return filename, "downloaded"
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(min(2**attempt, 8))
    return filename, f"error: {last_error}"


def download_many(filenames: Iterable[str], destination: Path, workers: int, retries: int) -> dict[str, int]:
    summary = {"downloaded": 0, "cached": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, filename, destination, retries): filename
            for filename in filenames
        }
        for future in as_completed(futures):
            filename, status = future.result()
            bucket = status if status in summary else "error"
            summary[bucket] += 1
            if bucket == "error":
                print(f"{filename}: {status}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revis-root", default="downloads/hallucination_refs/REVIS")
    parser.add_argument("--max-calibration-samples", type=int, default=300)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.revis_root)
    extraction = extraction_filenames(root / "data/OH_sampled_data_100.json")
    calibration = calibration_filenames(
        root / "data/coco_pope_calibration_merged.jsonl",
        args.max_calibration_samples,
    )
    extraction_dir = root / "data/OH_sampled_images_100"
    calibration_dir = root / "data/coco_pope_train_risk/mini_train2014"

    extraction_summary = download_many(extraction, extraction_dir, args.workers, args.retries)
    calibration_summary = download_many(calibration, calibration_dir, args.workers, args.retries)
    result = {
        "source": COCO_TRAIN2014,
        "extraction_samples": len(extraction),
        "calibration_rows": args.max_calibration_samples,
        "calibration_unique_images": len(calibration),
        "extraction": extraction_summary,
        "calibration": calibration_summary,
    }
    print(json.dumps(result, indent=2))
    return 1 if extraction_summary["error"] or calibration_summary["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
