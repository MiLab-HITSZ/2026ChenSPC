import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from analyze_cprc_robustness import (
    compact_trial,
    load_model_rows,
    run_frozen,
    summarize_trials,
)


VERSION = "cprc_calibration_size_v1"


def development_pair_order(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> Dict[str, List[str]]:
    by_subcategory: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not str(row.get("_dataset", "")).startswith("dev_"):
            continue
        by_subcategory[str(row.get("subcategory"))].add(str(row.get("pair_id")))
    if not by_subcategory:
        raise ValueError("no development rows")

    rng = random.Random(int(seed))
    order: Dict[str, List[str]] = {}
    for subcategory in sorted(by_subcategory):
        candidates = sorted(by_subcategory[subcategory])
        rng.shuffle(candidates)
        order[subcategory] = candidates
    return order


def selected_manifest(
    order: Mapping[str, Sequence[str]], pairs_per_subcategory: int
) -> Dict[str, List[str]]:
    selected = {
        subcategory: sorted(pair_ids[:pairs_per_subcategory])
        for subcategory, pair_ids in order.items()
    }
    if any(len(pair_ids) != pairs_per_subcategory for pair_ids in selected.values()):
        raise ValueError("requested calibration size exceeds the development pool")
    return selected


def restrict_development_rows(
    rows: Sequence[Dict[str, Any]], selected: Mapping[str, Sequence[str]]
) -> List[Dict[str, Any]]:
    selected_pairs = {
        (subcategory, pair_id)
        for subcategory, pair_ids in selected.items()
        for pair_id in pair_ids
    }
    output = []
    for row in rows:
        dataset = str(row.get("_dataset", ""))
        if dataset.startswith("test_"):
            output.append(row)
        elif dataset.startswith("dev_") and (
            str(row.get("subcategory")), str(row.get("pair_id"))
        ) in selected_pairs:
            output.append(row)
    return output


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibration_size_curve(
    model: str,
    rows: Sequence[Dict[str, Any]],
    settings: Mapping[str, Any],
    subset_seeds: Sequence[int],
    pairs_per_subcategory: Sequence[int],
    fit_seed: int,
) -> Dict[str, Any]:
    trials_by_size: Dict[str, List[Dict[str, Any]]] = {
        str(value): [] for value in pairs_per_subcategory
    }
    manifests = []
    result_cache: Dict[str, Dict[str, Any]] = {}

    for subset_seed in subset_seeds:
        order = development_pair_order(rows, int(subset_seed))
        previous_pairs: set[tuple[str, str]] = set()
        for per_subcategory in pairs_per_subcategory:
            selected = selected_manifest(order, int(per_subcategory))
            selected_pairs = {
                (subcategory, pair_id)
                for subcategory, pair_ids in selected.items()
                for pair_id in pair_ids
            }
            if not previous_pairs.issubset(selected_pairs):
                raise AssertionError("calibration subsets must be nested within each seed")
            previous_pairs = selected_pairs

            manifest_hash = canonical_hash(selected)
            if manifest_hash not in result_cache:
                subset_rows = restrict_development_rows(rows, selected)
                result_cache[manifest_hash] = run_frozen(
                    subset_rows, settings, int(fit_seed)
                )
            result = result_cache[manifest_hash]
            trial = compact_trial(
                restrict_development_rows(rows, selected), result, int(subset_seed)
            )
            trial.pop("predictions", None)
            trial["development_pairs"] = len(selected_pairs)
            trial["pairs_per_subcategory"] = int(per_subcategory)
            trial["manifest_sha256"] = manifest_hash
            trials_by_size[str(per_subcategory)].append(trial)
            manifests.append(
                {
                    "subset_seed": int(subset_seed),
                    "pairs_per_subcategory": int(per_subcategory),
                    "development_pairs": len(selected_pairs),
                    "sha256": manifest_hash,
                    "selected": selected,
                }
            )

    return {
        "model": model,
        "fit_seed": int(fit_seed),
        "subset_seeds": [int(seed) for seed in subset_seeds],
        "trials_by_pairs_per_subcategory": trials_by_size,
        "summary_by_pairs_per_subcategory": {
            key: summarize_trials(trials) for key, trials in trials_by_size.items()
        },
        "manifests": manifests,
        "selection_rule": (
            "For each seed, development pairs are randomly ordered within each of the "
            "14 subcategories. The 14/28/42/56/70-pair subsets are nested; both image "
            "sides and both answer formats stay grouped. The 230-pair test set never "
            "participates in fitting or hyperparameter selection."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure BPRC sensitivity to the number of calibration pairs."
    )
    parser.add_argument("--config", default="configs/cprc_calibration_size_v1.json")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    requested = set(args.model or config["models"])
    unknown = requested - set(config["models"])
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")

    output: Dict[str, Any] = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "pairs_per_subcategory": config["pairs_per_subcategory"],
        "experiments": {},
    }
    for model, spec in config["models"].items():
        if model not in requested:
            continue
        rows = load_model_rows(spec)
        output["experiments"][model] = calibration_size_curve(
            model,
            rows,
            config["fit"],
            config["subset_seeds"],
            config["pairs_per_subcategory"],
            int(config["fit_seed"]),
        )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("model\tpairs\ttask\tdCF_mean\tdCF_std\tdCS_mean\tdCS_std")
    for model, experiment in output["experiments"].items():
        for per_subcategory, summary in experiment[
            "summary_by_pairs_per_subcategory"
        ].items():
            pairs = 14 * int(per_subcategory)
            for task in ("mc", "qa"):
                cf = summary[task]["counterfactual"]["delta"]
                cs = summary[task]["commonsense"]["delta"]
                print(
                    f"{model}\t{pairs}\t{task}\t{cf['mean']:+.4f}\t{cf['std']:.4f}"
                    f"\t{cs['mean']:+.4f}\t{cs['std']:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
