import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


METHOD_KEYWORDS = (
    "bayes",
    "caption",
    "causal",
    "cp_vbc",
    "gate_",
    "jspace",
    "mfcd",
    "mitigation",
    "native_mc",
    "native_qa",
    "option_entailment",
    "prior_claim",
    "revis",
    "scope",
    "temporal",
    "vcd",
    "visual_claim",
)

CPR_KEYWORDS = (
    "bayes_path",
    "cp_vbc",
    "hierarchical_eb",
    "native_mc",
    "native_qa",
)

EXTERNAL_KEYWORDS = ("hallusion", "popev2", "pope_")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        lines = path.open("r", encoding="utf-8", errors="ignore")
    except OSError:
        return
    with lines:
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def artifact_family(path: Path) -> str:
    lowered = str(path).lower()
    if any(value in lowered for value in EXTERNAL_KEYWORDS):
        return "external"
    if any(value in lowered for value in CPR_KEYWORDS):
        return "cpr_development"
    if any(value in lowered for value in METHOD_KEYWORDS):
        return "other_method_development"
    return "baseline_evaluation"


def is_cdh_row(
    row: Dict[str, Any],
    pair_ids: Set[str],
    subcategories: Set[str],
    artifact: Path,
) -> bool:
    pair_id = str(row.get("pair_id") or "")
    if pair_id not in pair_ids or artifact_family(artifact) == "external":
        return False
    image_path = str(row.get("image_path") or "").replace("\\", "/")
    if image_path:
        return image_path.startswith("images/") or "/images/" in image_path
    return str(row.get("subcategory") or "") in subcategories


def stable_manifest_hash(pair_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(pair_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_ledger(benchmark: Path, result_root: Path) -> Dict[str, Any]:
    benchmark_rows = list(read_jsonl(benchmark))
    metadata = {
        str(row["pair_id"]): {
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "pair_name": row.get("pair_name"),
        }
        for row in benchmark_rows
    }
    pair_ids = set(metadata)
    subcategories = {
        str(row.get("subcategory") or "") for row in benchmark_rows
    }

    pair_artifacts: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    pair_tracks: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    artifact_rows: Dict[str, int] = Counter()
    artifact_pairs: Dict[str, Set[str]] = defaultdict(set)
    scanned_jsonl = 0

    for artifact in sorted(result_root.rglob("*.jsonl")):
        scanned_jsonl += 1
        family = artifact_family(artifact)
        relative = str(artifact)
        for row in read_jsonl(artifact):
            if not is_cdh_row(row, pair_ids, subcategories, artifact):
                continue
            pair_id = str(row["pair_id"])
            task = str(row.get("task") or "unknown")
            pair_artifacts[pair_id][family].add(relative)
            pair_tracks[pair_id][family].add(task)
            artifact_rows[relative] += 1
            artifact_pairs[relative].add(pair_id)

    baseline_exposed = {
        pair_id
        for pair_id in pair_ids
        if pair_artifacts[pair_id].get("baseline_evaluation")
    }
    method_exposed = {
        pair_id
        for pair_id in pair_ids
        if pair_artifacts[pair_id].get("cpr_development")
        or pair_artifacts[pair_id].get("other_method_development")
    }
    cpr_exposed = {
        pair_id
        for pair_id in pair_ids
        if pair_artifacts[pair_id].get("cpr_development")
    }
    mitigation_unseen = pair_ids - method_exposed
    cpr_unseen = pair_ids - cpr_exposed

    rows: List[Dict[str, Any]] = []
    for pair_id in sorted(
        pair_ids, key=lambda value: int(value.split()[-1])
    ):
        families = pair_artifacts[pair_id]
        family_summary = {}
        for family, paths in sorted(families.items()):
            ordered = sorted(paths)
            family_summary[family] = {
                "artifact_count": len(ordered),
                "tracks": sorted(pair_tracks[pair_id][family]),
                "artifact_examples": ordered[:20],
                "artifact_examples_truncated": len(ordered) > 20,
            }
        rows.append(
            {
                "pair_id": pair_id,
                **metadata[pair_id],
                "baseline_exposed": pair_id in baseline_exposed,
                "method_exposed": pair_id in method_exposed,
                "cpr_exposed": pair_id in cpr_exposed,
                "families": family_summary,
            }
        )

    by_subcategory = {}
    for subcategory in sorted(subcategories):
        selected = {
            pair_id
            for pair_id, value in metadata.items()
            if value["subcategory"] == subcategory
        }
        by_subcategory[subcategory] = {
            "total": len(selected),
            "baseline_exposed": len(selected & baseline_exposed),
            "method_exposed": len(selected & method_exposed),
            "cpr_exposed": len(selected & cpr_exposed),
            "mitigation_unseen": len(selected & mitigation_unseen),
            "cpr_unseen": len(selected & cpr_unseen),
        }

    artifacts = [
        {
            "path": path,
            "family": artifact_family(Path(path)),
            "row_count": artifact_rows[path],
            "pair_count": len(artifact_pairs[path]),
        }
        for path in sorted(artifact_rows)
    ]
    return {
        "version": "cdh_exposure_ledger_v1",
        "benchmark": str(benchmark),
        "result_root": str(result_root),
        "definitions": {
            "benchmark_unseen": (
                "No prior benchmark evaluation or method artifact contains the pair."
            ),
            "mitigation_unseen": (
                "The old benchmark baseline may contain the pair, but no detected "
                "mitigation or causal-diagnostic artifact contains it."
            ),
            "cpr_unseen": (
                "No detected CPR/CP-VBC/BayesPath/native candidate artifact contains it."
            ),
            "warning": (
                "A pair from an already published benchmark is not distribution-unseen "
                "even when it is mitigation-unseen."
            ),
        },
        "summary": {
            "benchmark_pairs": len(pair_ids),
            "scanned_jsonl_files": scanned_jsonl,
            "baseline_exposed_pairs": len(baseline_exposed),
            "method_exposed_pairs": len(method_exposed),
            "cpr_exposed_pairs": len(cpr_exposed),
            "benchmark_unseen_pairs": len(pair_ids - baseline_exposed - method_exposed),
            "mitigation_unseen_pairs": len(mitigation_unseen),
            "cpr_unseen_pairs": len(cpr_unseen),
        },
        "split_manifests": {
            "method_development": {
                "pair_ids": sorted(method_exposed),
                "sha256": stable_manifest_hash(method_exposed),
            },
            "mitigation_unseen": {
                "pair_ids": sorted(mitigation_unseen),
                "sha256": stable_manifest_hash(mitigation_unseen),
            },
            "cpr_development": {
                "pair_ids": sorted(cpr_exposed),
                "sha256": stable_manifest_hash(cpr_exposed),
            },
            "cpr_unseen": {
                "pair_ids": sorted(cpr_unseen),
                "sha256": stable_manifest_hash(cpr_unseen),
            },
        },
        "by_subcategory": by_subcategory,
        "pairs": rows,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a CDH pair-exposure ledger from saved JSONL artifacts."
    )
    parser.add_argument(
        "--benchmark", default="CDH-Bench.revised.strict.jsonl"
    )
    parser.add_argument("--result-root", default="result")
    parser.add_argument(
        "--output", default="runtime/paper/cdh_exposure_ledger.json"
    )
    args = parser.parse_args()

    output = build_ledger(Path(args.benchmark), Path(args.result_root))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
