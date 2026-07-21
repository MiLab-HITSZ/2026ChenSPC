import argparse
import copy
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_hierarchical_eb_lambda_cv import (
    candidate_arrays,
    exact_mcnemar_pvalue,
    resolved_baseline_key,
    run_cv,
)


VERSION = "spc_robustness_v1"
SIDES = ("counterfactual", "commonsense")


def load_model_rows(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    inputs = (
        ("dev_mc", "mc", spec["dev_mc"]),
        ("dev_qa", "qa", spec["dev_qa"]),
        ("test_mc", "mc", spec["test"]),
        ("test_qa", "qa", spec["test"]),
    )
    seen = set()
    for label, task, path in inputs:
        for source_row in read_jsonl(Path(path), task):
            row = copy.deepcopy(source_row)
            key = (task, str(row.get("pair_id")), str(row.get("side")))
            if key in seen:
                raise ValueError(f"duplicate model row: {key}")
            seen.add(key)
            row["_dataset"] = label
            row["_task"] = task
            rows.append(row)
    return rows


def row_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("_task") or row.get("task") or ""),
        str(row.get("pair_id") or ""),
        str(row.get("side") or ""),
    )


def candidate_signature(row: Mapping[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    keys, _, _ = candidate_arrays(dict(row))
    return str(row.get("_task") or row.get("task") or ""), tuple(keys)


def prior_by_key(row: Mapping[str, Any]) -> Dict[str, float]:
    keys, _, prior = candidate_arrays(dict(row))
    return dict(zip(keys, prior.tolist()))


def replace_prior(row: Dict[str, Any], values: Mapping[str, float]) -> None:
    candidates = row["cp_vbc"]["candidates"]
    for candidate in candidates:
        key = normalize_key(candidate["key"])
        candidate["logp_prior"] = float(values[key])


def development_ids(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    return [
        index
        for index, row in enumerate(rows)
        if str(row.get("_dataset", "")).startswith("dev_")
    ]


def mean_prior_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transformed = copy.deepcopy(list(rows))
    pools: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, float]]] = defaultdict(list)
    for index in development_ids(rows):
        pools[candidate_signature(rows[index])].append(prior_by_key(rows[index]))
    means = {
        signature: {
            key: float(np.mean([values[key] for values in pool]))
            for key in signature[1]
        }
        for signature, pool in pools.items()
    }
    for row in transformed:
        replace_prior(row, means[candidate_signature(row)])
    return transformed


def shuffled_prior_rows(
    rows: Sequence[Dict[str, Any]], seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    transformed = copy.deepcopy(list(rows))
    pools: Dict[Tuple[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    for index in development_ids(rows):
        pools[candidate_signature(rows[index])].append(index)
    rng = random.Random(int(seed))
    donors: Dict[str, str] = {}
    for index, row in enumerate(transformed):
        pool = list(pools[candidate_signature(row)])
        alternatives = [donor for donor in pool if row_key(rows[donor]) != row_key(row)]
        donor = rng.choice(alternatives or pool)
        replace_prior(row, prior_by_key(rows[donor]))
        donors["/".join(row_key(row))] = "/".join(row_key(rows[donor]))
    return transformed, donors


def derangement(size: int, rng: random.Random) -> List[int]:
    if size < 2:
        return list(range(size))
    values = list(range(size))
    for _ in range(100):
        rng.shuffle(values)
        if all(index != value for index, value in enumerate(values)):
            return list(values)
    shift = rng.randrange(1, size)
    return [(index + shift) % size for index in range(size)]


def permuted_prior_rows(
    rows: Sequence[Dict[str, Any]], seed: int
) -> List[Dict[str, Any]]:
    transformed = copy.deepcopy(list(rows))
    rng = random.Random(int(seed))
    for row in transformed:
        keys, _, prior = candidate_arrays(row)
        permutation = derangement(len(keys), rng)
        replace_prior(
            row,
            {key: float(prior[permutation[index]]) for index, key in enumerate(keys)},
        )
    return transformed


def image_residual_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transformed = copy.deepcopy(list(rows))
    for row in transformed:
        keys, image, _ = candidate_arrays(row)
        replace_prior(row, dict(zip(keys, image.tolist())))
    return transformed


def predictions_from_result(
    rows: Sequence[Dict[str, Any]], result: Mapping[str, Any]
) -> Dict[Tuple[str, str, str], str]:
    predictions = {
        row_key(row): resolved_baseline_key(row)
        for row in rows
        if str(row.get("_dataset", "")).startswith("test_")
    }
    for case in result["cases"]:
        if str(case.get("dataset", "")).startswith("test_"):
            predictions[
                (str(case["task"]), str(case["pair_id"]), str(case["side"]))
            ] = normalize_key(case["prediction"])
    return predictions


def side_outcomes(
    rows: Sequence[Dict[str, Any]],
    predictions: Mapping[Tuple[str, str, str], str],
    task: str,
    side: str,
) -> List[int]:
    selected = sorted(
        (
            row
            for row in rows
            if row.get("_task") == task
            and row.get("side") == side
            and str(row.get("_dataset", "")).startswith("test_")
        ),
        key=row_key,
    )
    return [
        int(predictions[row_key(row)] == normalize_key(row.get("gt")))
        for row in selected
    ]


def paired_comparison(control: Sequence[int], full: Sequence[int]) -> Dict[str, Any]:
    if len(control) != len(full):
        raise ValueError("paired outcomes must have equal length")
    repairs = sum(not reference and candidate for candidate, reference in zip(control, full))
    harms = sum(reference and not candidate for candidate, reference in zip(control, full))
    return {
        "n": len(control),
        "delta_vs_full": (sum(control) - sum(full)) / len(full) if full else 0.0,
        "repairs_vs_full": repairs,
        "harms_vs_full": harms,
        "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(repairs, harms),
    }


def run_frozen(
    rows: Sequence[Dict[str, Any]],
    settings: Mapping[str, Any],
    seed: int,
    feature_mode: str = "full",
) -> Dict[str, Any]:
    return run_cv(
        rows,
        "frozen_transfer",
        5,
        float(settings["max_cs_drop"]),
        True,
        int(settings["quadrature_points"]),
        int(settings["bootstrap"]),
        int(seed),
        "net_utility",
        float(settings["cs_cost"]),
        ("dev_mc", "dev_qa"),
        feature_mode,
    )


def compact_trial(
    rows: Sequence[Dict[str, Any]], result: Mapping[str, Any], seed: int
) -> Dict[str, Any]:
    predictions = predictions_from_result(rows, result)
    tasks = {}
    for task in ("mc", "qa"):
        label = f"test_{task}"
        metrics = result["evaluation_by_dataset"][label]
        tests = result["paired_tests"][label]
        cases = result["case_summary"][label]
        tasks[task] = {
            side: {
                **metrics[side],
                **tests[side],
                "intervention_rate": cases[side]["overrides"] / metrics[side]["n"],
            }
            for side in SIDES
        }
        tasks[task]["cfad"] = (
            metrics["commonsense"]["acc"] - metrics["counterfactual"]["acc"]
        )
        tasks[task]["rpd"] = (
            tasks[task]["cfad"] / metrics["commonsense"]["acc"]
            if metrics["commonsense"]["acc"]
            else None
        )
    return {
        "seed": int(seed),
        "tasks": tasks,
        "predictions": {"/".join(key): value for key, value in predictions.items()},
        "selections": result["selections"],
    }


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    numbers = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def summarize_trials(trials: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        task: {
            side: {
                field: mean_std(
                    trial["tasks"][task][side][field] for trial in trials
                )
                for field in ("acc", "delta", "intervention_rate")
            }
            for side in SIDES
        }
        for task in ("mc", "qa")
    }


def prior_controls(
    model: str,
    rows: Sequence[Dict[str, Any]],
    settings: Mapping[str, Any],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    full_result = run_frozen(rows, settings, int(seeds[0]))
    full = compact_trial(rows, full_result, int(seeds[0]))
    variants: Dict[str, List[Dict[str, Any]]] = {"full": [full]}

    deterministic = (
        ("development_mean_prior", mean_prior_rows(rows), "full"),
        ("image_residual", image_residual_rows(rows), "image_only"),
    )
    for name, transformed, feature_mode in deterministic:
        result = run_frozen(transformed, settings, int(seeds[0]), feature_mode)
        variants[name] = [compact_trial(transformed, result, int(seeds[0]))]

    for name in ("development_shuffled_prior", "within_row_permuted_prior"):
        variants[name] = []
        for seed in seeds:
            if name == "development_shuffled_prior":
                transformed, _ = shuffled_prior_rows(rows, int(seed))
            else:
                transformed = permuted_prior_rows(rows, int(seed))
            result = run_frozen(transformed, settings, int(seed))
            variants[name].append(compact_trial(transformed, result, int(seed)))

    full_predictions = {
        tuple(key.split("/", 2)): value for key, value in full["predictions"].items()
    }
    comparisons: Dict[str, Any] = {}
    for name, trials in variants.items():
        if name == "full":
            continue
        comparisons[name] = []
        for trial in trials:
            predictions = {
                tuple(key.split("/", 2)): value
                for key, value in trial["predictions"].items()
            }
            by_task = {}
            for task in ("mc", "qa"):
                by_task[task] = {}
                for side in SIDES:
                    control = side_outcomes(rows, predictions, task, side)
                    reference = side_outcomes(rows, full_predictions, task, side)
                    by_task[task][side] = paired_comparison(control, reference)
            comparisons[name].append({"seed": trial["seed"], "tasks": by_task})

    for trials in variants.values():
        for trial in trials:
            trial.pop("predictions", None)

    return {
        "model": model,
        "seed_count": len(seeds),
        "variants": variants,
        "summary": {name: summarize_trials(trials) for name, trials in variants.items()},
        "paired_vs_full": comparisons,
        "fairness": (
            "Every control uses the same development rows, EB dimension, regularization/cap/gate "
            "grid, CS constraint, and selection objective. Shuffled test priors are drawn only "
            "from the development pool; no test labels or test priors select a donor."
        ),
    }


def choose_development_pairs(
    rows: Sequence[Mapping[str, Any]], per_subcategory: int, seed: int
) -> Dict[str, List[str]]:
    by_subcategory: Dict[str, set] = defaultdict(set)
    for row in rows:
        by_subcategory[str(row.get("subcategory"))].add(str(row.get("pair_id")))
    rng = random.Random(int(seed))
    selected: Dict[str, List[str]] = {}
    for subcategory in sorted(by_subcategory):
        candidates = sorted(by_subcategory[subcategory])
        if len(candidates) < per_subcategory:
            raise ValueError(f"not enough pairs in {subcategory}: {len(candidates)}")
        selected[subcategory] = sorted(rng.sample(candidates, per_subcategory))
    return selected


def relabel_split(
    rows: Sequence[Dict[str, Any]], selected: Mapping[str, Sequence[str]]
) -> List[Dict[str, Any]]:
    development = {
        (subcategory, pair_id)
        for subcategory, pair_ids in selected.items()
        for pair_id in pair_ids
    }
    output = copy.deepcopy(list(rows))
    for row in output:
        prefix = (
            "dev"
            if (str(row.get("subcategory")), str(row.get("pair_id"))) in development
            else "test"
        )
        row["_dataset"] = f"{prefix}_{row['_task']}"
    return output


def repeated_splits(
    model: str,
    rows: Sequence[Dict[str, Any]],
    settings: Mapping[str, Any],
    seeds: Sequence[int],
    per_subcategory: int,
) -> Dict[str, Any]:
    trials = []
    manifests = []
    for seed in seeds:
        selected = choose_development_pairs(rows, per_subcategory, int(seed))
        canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"))
        manifests.append(
            {
                "seed": int(seed),
                "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "development_pairs": selected,
            }
        )
        split_rows = relabel_split(rows, selected)
        result = run_frozen(split_rows, settings, int(seed))
        trial = compact_trial(split_rows, result, int(seed))
        trial.pop("predictions", None)
        trials.append(trial)
    return {
        "model": model,
        "trials": trials,
        "summary": summarize_trials(trials),
        "split_manifests": manifests,
        "selection_rule": (
            f"Exactly {per_subcategory} pair IDs per subcategory, sampled from benchmark "
            "metadata with a frozen seed before refitting; both sides and both tasks remain grouped."
        ),
    }


def operating_points(
    model: str,
    rows: Sequence[Dict[str, Any]],
    settings: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    definitions = {
        "cs_safe": {"max_cs_drop": 0.05, "cs_cost": 0.5},
        "high_repair": {"max_cs_drop": 0.20, "cs_cost": 0.5},
    }
    output = {}
    for name, override in definitions.items():
        local = dict(settings)
        local.update(override)
        result = run_frozen(rows, local, seed)
        trial = compact_trial(rows, result, seed)
        trial.pop("predictions", None)
        output[name] = trial
    return {"model": model, "definitions": definitions, "results": output}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen SPC robustness experiments.")
    parser.add_argument("--config", default="configs/spc_robustness_v1.json")
    parser.add_argument(
        "--experiment",
        choices=("all", "prior_controls", "repeated_splits", "operating_points"),
        default="all",
    )
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    requested = set(args.model or config["models"])
    unknown = requested - set(config["models"])
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")

    result: Dict[str, Any] = {
        "version": VERSION,
        "config": str(args.config),
        "frozen_seeds": config["seeds"],
        "experiments": {},
    }
    for model in config["models"]:
        if model not in requested:
            continue
        rows = load_model_rows(config["models"][model])
        model_result: Dict[str, Any] = {}
        if args.experiment in {"all", "prior_controls"} and model in config["prior_control_models"]:
            model_result["prior_controls"] = prior_controls(
                model, rows, config["fit"], config["seeds"]["prior_controls"]
            )
        if args.experiment in {"all", "repeated_splits"}:
            model_result["repeated_splits"] = repeated_splits(
                model,
                rows,
                config["fit"],
                config["seeds"]["repeated_splits"],
                int(config["development_pairs_per_subcategory"]),
            )
        if args.experiment in {"all", "operating_points"}:
            model_result["operating_points"] = operating_points(
                model, rows, config["fit"], int(config["seeds"]["fit"])
            )
        result["experiments"][model] = model_result

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({model: list(value) for model, value in result["experiments"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
