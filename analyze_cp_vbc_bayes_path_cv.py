import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SIDES = ("counterfactual", "commonsense")


def read_jsonl(path: Path, task: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") != "ok" or not isinstance(row.get("cp_vbc"), dict):
            continue
        if task and row.get("task") != task:
            continue
        if row.get("side") in SIDES:
            rows.append(row)
    return rows


def parse_input(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be LABEL=RESULTS_JSONL")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--input label cannot be empty")
    return label, Path(raw_path).expanduser()


def normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


def resolved_baseline_key(row: Dict[str, Any]) -> Any:
    cp_vbc = row.get("cp_vbc") or {}
    identity = " ".join(
        str(value or "")
        for value in (
            row.get("model_name"),
            row.get("model"),
            cp_vbc.get("model"),
        )
    ).lower()
    baseline_pred = str(cp_vbc.get("baseline_pred") or "")
    if "thinking" in identity and "</think>" not in baseline_pred:
        return None
    return cp_vbc.get("baseline_key")


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def top_and_margin(scores: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
    if not scores:
        return None, None
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) == 1:
        return ranked[0][0], None
    return ranked[0][0], float(ranked[0][1] - ranked[1][1])


def candidate_scores(row: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, float], Dict[str, float]]:
    cp = row["cp_vbc"]
    image_scores: Dict[str, float] = {}
    prior_scores: Dict[str, float] = {}
    for candidate in cp.get("candidates") or []:
        key = str(candidate["key"])
        image_scores[key] = float(candidate["logp_image"])
        prior_scores[key] = float(candidate["logp_prior"])
    return resolved_baseline_key(row), image_scores, prior_scores


def fixed_features(row: Dict[str, Any], lambda_text: float) -> Dict[str, Any]:
    baseline, image_scores, prior_scores = candidate_scores(row)
    contrast_scores = {
        key: image_scores[key] - float(lambda_text) * prior_scores[key]
        for key in image_scores
    }
    image_top, image_margin = top_and_margin(image_scores)
    prior_top, prior_margin = top_and_margin(prior_scores)
    proposal, contrast_margin = top_and_margin(contrast_scores)
    prior_relief = None
    if baseline in prior_scores and proposal in prior_scores:
        prior_relief = prior_scores[baseline] - prior_scores[proposal]
    return {
        "baseline": baseline,
        "proposal": proposal,
        "image_top": image_top,
        "prior_top": prior_top,
        "image_margin": image_margin,
        "prior_margin": prior_margin,
        "contrast_margin": contrast_margin,
        "prior_relief": prior_relief,
        "visual_conflict": bool(image_top and prior_top and proposal and image_top != prior_top and proposal == image_top),
        "prior_absorption": bool(
            baseline
            and image_top
            and prior_top
            and proposal
            and baseline == image_top == prior_top
            and proposal != baseline
        ),
    }


def bayes_path_features(
    row: Dict[str, Any],
    lambda_low: float,
    lambda_high: float,
    steps: int,
) -> Dict[str, Any]:
    baseline, image_scores, prior_scores = candidate_scores(row)
    keys = list(image_scores)
    image_top, image_margin = top_and_margin(image_scores)
    prior_top, prior_margin = top_and_margin(prior_scores)
    posterior_mean = {key: 0.0 for key in keys}
    top_counts = {key: 0 for key in keys}
    for step in range(max(2, int(steps))):
        weight = step / (max(2, int(steps)) - 1)
        lambda_text = float(lambda_low) + (float(lambda_high) - float(lambda_low)) * weight
        scores = [image_scores[key] - lambda_text * prior_scores[key] for key in keys]
        probabilities = softmax(scores)
        top_key = keys[max(range(len(keys)), key=lambda idx: probabilities[idx])]
        top_counts[top_key] += 1
        for key, probability in zip(keys, probabilities):
            posterior_mean[key] += probability / max(2, int(steps))

    proposal, path_margin = top_and_margin(posterior_mean)
    stability = (top_counts.get(proposal, 0) / max(2, int(steps))) if proposal else 0.0
    prior_relief = None
    if baseline in prior_scores and proposal in prior_scores:
        prior_relief = prior_scores[baseline] - prior_scores[proposal]
    return {
        "baseline": baseline,
        "proposal": proposal,
        "image_top": image_top,
        "prior_top": prior_top,
        "image_margin": image_margin,
        "prior_margin": prior_margin,
        "path_margin": path_margin,
        "path_stability": stability,
        "prior_relief": prior_relief,
        "posterior_mean": posterior_mean,
        "visual_conflict": bool(image_top and prior_top and proposal and image_top != prior_top and proposal == image_top),
        "prior_absorption": bool(
            baseline
            and image_top
            and prior_top
            and proposal
            and baseline == image_top == prior_top
            and proposal != baseline
        ),
    }


def relief_passes(value: Optional[float], threshold: Optional[float], baseline: Optional[str]) -> bool:
    if baseline is None or threshold is None:
        return True
    return value is not None and value >= threshold


def predict_fixed(features: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    image_margin = features["image_margin"]
    contrast_margin = features["contrast_margin"]
    visual_risk = features["visual_conflict"] and (
        image_margin is None or image_margin >= params["visual_margin_min"]
    )
    absorption_risk = features["prior_absorption"] and (
        image_margin is None or image_margin <= params["absorption_margin_max"]
    )
    enough_contrast = contrast_margin is None or contrast_margin >= params["contrast_margin_min"]
    enough_relief = relief_passes(
        features["prior_relief"], params["prior_relief_min"], features["baseline"]
    )
    if (visual_risk or absorption_risk) and enough_contrast and enough_relief:
        return features["proposal"]
    return features["baseline"]


def bayes_path_gate(features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    image_margin = features["image_margin"]
    path_margin = features["path_margin"]
    absorption_path_margin_max = params.get("absorption_path_margin_max")
    visual_risk = features["visual_conflict"] and (
        image_margin is None or image_margin >= params["visual_margin_min"]
    )
    absorption_risk = features["prior_absorption"] and (
        image_margin is None or image_margin <= params["absorption_margin_max"]
    ) and (
        absorption_path_margin_max is None
        or float(absorption_path_margin_max) < 0
        or path_margin is None
        or path_margin <= float(absorption_path_margin_max)
    )
    enough_relief = relief_passes(
        features["prior_relief"], params["prior_relief_min"], features["baseline"]
    )
    accepted = bool(
        (visual_risk or absorption_risk)
        and enough_relief
        and features["path_stability"] >= params["path_stability_min"]
        and (path_margin is None or path_margin >= params["path_margin_min"])
    )
    return {
        "accepted": accepted,
        "visual_risk": bool(visual_risk),
        "absorption_risk": bool(absorption_risk),
        "risk_type": (
            "visual_conflict"
            if visual_risk
            else "prior_absorption"
            if absorption_risk
            else None
        ),
        "enough_relief": bool(enough_relief),
    }


def predict_bayes_path(features: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    if bayes_path_gate(features, params)["accepted"]:
        return features["proposal"]
    return features["baseline"]


def fold_ids(
    rows: Sequence[Dict[str, Any]], requested_folds: Optional[int] = None
) -> List[int]:
    if requested_folds is not None and int(requested_folds) < 2:
        raise ValueError("requested_folds must be at least 2")
    grouped: Dict[str, set] = defaultdict(set)
    for row in rows:
        group = str(row.get("cv_group") or row.get("pair_id") or "")
        grouped[str(row.get("subcategory") or "")].add(group)
    lookup: Dict[Tuple[str, str], int] = {}
    for subcategory, pairs in grouped.items():
        ordered = sorted(
            pairs,
            key=lambda pair: (
                0,
                int(re.search(r"\d+", pair).group()),
            )
            if re.search(r"\d+", pair)
            else (1, pair),
        )
        if requested_folds is not None and len(ordered) < int(requested_folds):
            raise ValueError(
                f"subcategory {subcategory!r} has {len(ordered)} pairs, fewer than "
                f"the requested {requested_folds} folds"
            )
        for index, group in enumerate(ordered):
            lookup[(subcategory, group)] = (
                index % int(requested_folds) if requested_folds is not None else index
            )
    if requested_folds is not None:
        return [
            lookup[
                (
                    str(row.get("subcategory") or ""),
                    str(row.get("cv_group") or row.get("pair_id") or ""),
                )
            ]
            for row in rows
        ]
    counts = {len(pairs) for pairs in grouped.values()}
    if len(counts) != 1:
        raise ValueError(f"all subcategories must have the same pair count, got {sorted(counts)}")
    return [
        lookup[
            (
                str(row.get("subcategory") or ""),
                str(row.get("cv_group") or row.get("pair_id") or ""),
            )
        ]
        for row in rows
    ]


def metric(rows: Sequence[Dict[str, Any]], predictions: Sequence[Optional[str]], ids: Iterable[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    id_list = list(ids)
    for side in SIDES:
        selected = [index for index in id_list if rows[index].get("side") == side]
        baseline_correct = sum(
            normalize_key(resolved_baseline_key(rows[index]))
            == normalize_key(rows[index].get("gt"))
            for index in selected
        )
        correct = sum(
            normalize_key(predictions[index]) == normalize_key(rows[index].get("gt"))
            for index in selected
        )
        overrides = sum(
            normalize_key(predictions[index])
            != normalize_key(resolved_baseline_key(rows[index]))
            for index in selected
        )
        out[side] = {
            "n": len(selected),
            "baseline_correct": baseline_correct,
            "correct": correct,
            "baseline_acc": baseline_correct / len(selected) if selected else None,
            "acc": correct / len(selected) if selected else None,
            "delta": (correct - baseline_correct) / len(selected) if selected else None,
            "overrides": overrides,
        }
    return out


def metric_by_field(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Optional[str]],
    ids: Iterable[int],
    field: str,
) -> Dict[str, Any]:
    id_list = list(ids)
    keys = sorted({str(rows[index].get(field) or "") for index in id_list})
    return {
        key: metric(
            rows,
            predictions,
            [index for index in id_list if str(rows[index].get(field) or "") == key],
        )
        for key in keys
    }


def bootstrap_delta_ci(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Optional[str]],
    side: str,
    samples: int,
    seed: int,
) -> List[float]:
    ids = [index for index, row in enumerate(rows) if row.get("side") == side]
    differences = [
        int(normalize_key(predictions[index]) == normalize_key(rows[index].get("gt")))
        - int(
            normalize_key(resolved_baseline_key(rows[index]))
            == normalize_key(rows[index].get("gt"))
        )
        for index in ids
    ]
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(max(1, samples)):
        draws.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    draws.sort()
    lo = draws[max(0, int(0.025 * (len(draws) - 1)))]
    hi = draws[min(len(draws) - 1, int(0.975 * (len(draws) - 1)))]
    return [float(lo), float(hi)]


def selection_key(
    metrics: Dict[str, Any],
    total_overrides: int,
    params_index: int,
    selection_objective: str = "cf_first",
    cs_cost: float = 1.0,
) -> Tuple[float, ...]:
    cf = metrics["counterfactual"]
    cs = metrics["commonsense"]
    if selection_objective == "net_utility":
        utility = float(cf["delta"]) + float(cs_cost) * float(cs["delta"])
        return (
            utility,
            float(cf["delta"]),
            float(cs["delta"]),
            -int(cs["overrides"]),
            -int(total_overrides),
            -int(params_index),
        )
    return (
        int(cf["correct"]),
        int(cs["correct"]),
        -int(cs["overrides"]),
        -int(total_overrides),
        -int(params_index),
    )


def select_params(
    rows: Sequence[Dict[str, Any]],
    prediction_bank: Sequence[Sequence[Optional[str]]],
    params: Sequence[Dict[str, Any]],
    train_ids: Sequence[int],
    max_cs_drop: float,
    selection_objective: str = "cf_first",
    cs_cost: float = 1.0,
) -> int:
    best: Optional[Tuple[Tuple[float, ...], int]] = None
    for index, predictions in enumerate(prediction_bank):
        metrics = metric(rows, predictions, train_ids)
        if metrics["commonsense"]["delta"] < -abs(max_cs_drop) - 1e-12:
            continue
        total_overrides = metrics["counterfactual"]["overrides"] + metrics["commonsense"]["overrides"]
        key = selection_key(
            metrics,
            total_overrides,
            index,
            selection_objective=selection_objective,
            cs_cost=cs_cost,
        )
        if best is None or key > best[0]:
            best = (key, index)
    if best is None:
        raise RuntimeError("no parameter setting passed the CS preservation constraint")
    return best[1]


def fixed_grid() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lambda_text in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        for contrast_margin in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
            for absorption_margin in (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0):
                for visual_margin in (0.0, 0.5, 1.0):
                    for prior_relief in (None, 0.0, 0.25, 1.0):
                        rows.append(
                            {
                                "family": "fixed",
                                "lambda": lambda_text,
                                "contrast_margin_min": contrast_margin,
                                "absorption_margin_max": absorption_margin,
                                "visual_margin_min": visual_margin,
                                "prior_relief_min": prior_relief,
                            }
                        )
    return rows


def bayes_path_grid(steps: int) -> List[Dict[str, Any]]:
    intervals = (
        (0.2, 0.8),
        (0.3, 0.9),
        (0.4, 1.0),
        (0.4, 1.2),
        (0.5, 1.1),
        (0.6, 1.2),
        (0.8, 1.2),
        (0.5, 1.5),
    )
    rows: List[Dict[str, Any]] = []
    for lambda_low, lambda_high in intervals:
        for stability in (0.35, 0.5, 0.65, 0.8, 0.9):
            for path_margin in (0.0, 0.05, 0.1, 0.2, 0.3):
                for absorption_margin in (0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0):
                    for visual_margin in (0.0, 0.5, 1.0):
                        for prior_relief in (None, 0.0, 0.25, 1.0):
                            rows.append(
                                {
                                    "family": "bayes_path",
                                    "lambda_low": lambda_low,
                                    "lambda_high": lambda_high,
                                    "lambda_steps": steps,
                                    "path_stability_min": stability,
                                    "path_margin_min": path_margin,
                                    "absorption_margin_max": absorption_margin,
                                    "visual_margin_min": visual_margin,
                                    "prior_relief_min": prior_relief,
                                }
                            )
    return rows


def evaluate_dataset(
    label: str,
    path: Path,
    task: Optional[str],
    max_cs_drop: float,
    path_steps: int,
    bootstrap_samples: int,
    seed: int,
    requested_folds: Optional[int] = None,
    fold_mode: str = "group",
    selection_objective: str = "cf_first",
    cs_cost: float = 1.0,
) -> Dict[str, Any]:
    rows = read_jsonl(path, task)
    if not rows:
        raise ValueError(f"no usable CP-VBC rows in {path}")
    if fold_mode == "subcategory":
        subcategories = sorted({str(row.get("subcategory") or "") for row in rows})
        lookup = {subcategory: index for index, subcategory in enumerate(subcategories)}
        folds = [lookup[str(row.get("subcategory") or "")] for row in rows]
    else:
        folds = fold_ids(rows, requested_folds=requested_folds)
    fold_count = max(folds) + 1
    all_ids = list(range(len(rows)))
    baseline_predictions = [resolved_baseline_key(row) for row in rows]
    saved_predictions = [row["cp_vbc"].get("final_key") for row in rows]

    fixed_params = fixed_grid()
    fixed_cache = {
        value: [fixed_features(row, value) for row in rows]
        for value in sorted({float(params["lambda"]) for params in fixed_params})
    }
    fixed_predictions = [
        [predict_fixed(features, params) for features in fixed_cache[float(params["lambda"])]]
        for params in fixed_params
    ]

    path_params = bayes_path_grid(path_steps)
    path_cache: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for params in path_params:
        interval = (float(params["lambda_low"]), float(params["lambda_high"]))
        if interval not in path_cache:
            path_cache[interval] = [
                bayes_path_features(row, interval[0], interval[1], path_steps) for row in rows
            ]
    path_predictions = [
        [
            predict_bayes_path(features, params)
            for features in path_cache[(float(params["lambda_low"]), float(params["lambda_high"]))]
        ]
        for params in path_params
    ]

    families = {
        "fixed": (fixed_params, fixed_predictions),
        "bayes_path": (path_params, path_predictions),
    }
    family_results: Dict[str, Any] = {}
    combined_oof = list(baseline_predictions)
    combined_selection: List[Dict[str, Any]] = []

    for family, (params, prediction_bank) in families.items():
        oof = list(baseline_predictions)
        fold_selection: List[Dict[str, Any]] = []
        for fold in range(fold_count):
            train_ids = [index for index, value in enumerate(folds) if value != fold]
            test_ids = [index for index, value in enumerate(folds) if value == fold]
            selected_index = select_params(
                rows,
                prediction_bank,
                params,
                train_ids,
                max_cs_drop,
                selection_objective=selection_objective,
                cs_cost=cs_cost,
            )
            for index in test_ids:
                oof[index] = prediction_bank[selected_index][index]
            fold_selection.append({"fold": fold, "params": params[selected_index]})

        full_index = select_params(
            rows,
            prediction_bank,
            params,
            all_ids,
            max_cs_drop,
            selection_objective=selection_objective,
            cs_cost=cs_cost,
        )
        serialized_selection = [
            json.dumps(item["params"], sort_keys=True, separators=(",", ":")) for item in fold_selection
        ]
        selection_counts = Counter(serialized_selection)
        consensus_serialized = max(
            selection_counts,
            key=lambda value: (selection_counts[value], -serialized_selection.index(value)),
        )
        consensus_params = json.loads(consensus_serialized)
        consensus_index = params.index(consensus_params)
        oof_metrics = metric(rows, oof, all_ids)
        family_results[family] = {
            "oof": oof_metrics,
            "by_subcategory_oof": metric_by_field(rows, oof, all_ids, "subcategory"),
            "oof_delta_ci95": {
                side: bootstrap_delta_ci(rows, oof, side, bootstrap_samples, seed + offset)
                for offset, side in enumerate(SIDES)
            },
            "fold_selection": fold_selection,
            "consensus_preset": {
                "params": consensus_params,
                "selected_in_folds": selection_counts[consensus_serialized],
                "metrics": metric(rows, prediction_bank[consensus_index], all_ids),
                "note": "modal outer-fold selection replayed on the full development set",
            },
            "full_data_selected": {
                "params": params[full_index],
                "metrics": metric(rows, prediction_bank[full_index], all_ids),
                "note": "development-set upper bound; not held out",
            },
            "_oof_predictions": oof,
        }

    combined_params = fixed_params + path_params
    combined_bank = fixed_predictions + path_predictions
    for fold in range(fold_count):
        train_ids = [index for index, value in enumerate(folds) if value != fold]
        test_ids = [index for index, value in enumerate(folds) if value == fold]
        selected_index = select_params(
            rows,
            combined_bank,
            combined_params,
            train_ids,
            max_cs_drop,
            selection_objective=selection_objective,
            cs_cost=cs_cost,
        )
        for index in test_ids:
            combined_oof[index] = combined_bank[selected_index][index]
        combined_selection.append({"fold": fold, "params": combined_params[selected_index]})

    cases: List[Dict[str, Any]] = []
    path_oof = family_results["bayes_path"]["_oof_predictions"]
    fixed_oof = family_results["fixed"]["_oof_predictions"]
    for index, row in enumerate(rows):
        baseline_correct = normalize_key(baseline_predictions[index]) == normalize_key(row.get("gt"))
        fixed_correct = normalize_key(fixed_oof[index]) == normalize_key(row.get("gt"))
        path_correct = normalize_key(path_oof[index]) == normalize_key(row.get("gt"))
        if fixed_oof[index] != path_oof[index] or baseline_correct != path_correct:
            cases.append(
                {
                    "side": row.get("side"),
                    "subcategory": row.get("subcategory"),
                    "pair_id": row.get("pair_id"),
                    "gt": row.get("gt"),
                    "baseline": baseline_predictions[index],
                    "fixed_oof": fixed_oof[index],
                    "bayes_path_oof": path_oof[index],
                    "baseline_correct": baseline_correct,
                    "fixed_correct": fixed_correct,
                    "bayes_path_correct": path_correct,
                }
            )

    for result in family_results.values():
        result.pop("_oof_predictions", None)
    return {
        "label": label,
        "path": str(path),
        "task": task or sorted({str(row.get("task")) for row in rows}),
        "n": len(rows),
        "folds": fold_count,
        "fold_mode": fold_mode,
        "selection_objective": selection_objective,
        "cs_cost": float(cs_cost),
        "baseline": metric(rows, baseline_predictions, all_ids),
        "by_subcategory_baseline": metric_by_field(
            rows, baseline_predictions, all_ids, "subcategory"
        ),
        "saved_cp_vbc": metric(rows, saved_predictions, all_ids),
        "families": family_results,
        "combined_oof": {
            "metrics": metric(rows, combined_oof, all_ids),
            "by_subcategory": metric_by_field(rows, combined_oof, all_ids, "subcategory"),
            "delta_ci95": {
                side: bootstrap_delta_ci(rows, combined_oof, side, bootstrap_samples, seed + 10 + offset)
                for offset, side in enumerate(SIDES)
            },
            "fold_selection": combined_selection,
        },
        "case_differences": cases,
    }


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grouped-CV comparison of fixed-lambda CP-VBC and Bayesian lambda-path marginalization."
    )
    parser.add_argument("--input", action="append", type=parse_input, required=True, help="LABEL=results.jsonl")
    parser.add_argument("--task", default="", help="Optional task filter")
    parser.add_argument("--max-cs-drop", type=float, default=0.03)
    parser.add_argument(
        "--selection-objective",
        choices=("cf_first", "net_utility"),
        default="cf_first",
        help=(
            "cf_first preserves the legacy lexicographic selector; net_utility "
            "maximizes delta_CF + cs_cost * delta_CS within max-cs-drop."
        ),
    )
    parser.add_argument(
        "--cs-cost",
        type=float,
        default=1.0,
        help="Cost assigned to one point of CS change under net_utility.",
    )
    parser.add_argument("--path-steps", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="Explicit grouped fold count. Default 0 preserves the legacy per-subcategory rank folds.",
    )
    parser.add_argument(
        "--fold-mode",
        choices=("group", "subcategory"),
        default="group",
        help="group keeps cv_group/pair groups intact; subcategory performs LOSO.",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    datasets = [
        evaluate_dataset(
            label,
            path,
            args.task or None,
            args.max_cs_drop,
            max(2, args.path_steps),
            max(1, args.bootstrap_samples),
            args.seed + index * 100,
            args.folds if args.folds > 0 else None,
            args.fold_mode,
            args.selection_objective,
            max(0.0, float(args.cs_cost)),
        )
        for index, (label, path) in enumerate(args.input)
    ]
    output = {
        "version": "cp_vbc_bayes_path_cv_v1",
        "max_cs_drop": args.max_cs_drop,
        "selection_objective": args.selection_objective,
        "cs_cost": max(0.0, float(args.cs_cost)),
        "fairness": (
            "cv_group (when present), pair identity, and subcategory are used only to form grouped folds. "
            "Final predictions use only the "
            "single image, question/answer candidates, and image/no-image candidate likelihoods."
        ),
        "datasets": datasets,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("dataset\tmethod\tCF_Acc\tdCF\tCS_Acc\tdCS\tCF_CI95\tCS_CI95")
    for dataset in datasets:
        for family in ("fixed", "bayes_path"):
            result = dataset["families"][family]
            metrics = result["oof"]
            print(
                "\t".join(
                    (
                        dataset["label"],
                        f"{family}_oof",
                        fmt(metrics["counterfactual"]["acc"]),
                        fmt(metrics["counterfactual"]["delta"]),
                        fmt(metrics["commonsense"]["acc"]),
                        fmt(metrics["commonsense"]["delta"]),
                        str(result["oof_delta_ci95"]["counterfactual"]),
                        str(result["oof_delta_ci95"]["commonsense"]),
                    )
                )
            )
        metrics = dataset["combined_oof"]["metrics"]
        print(
            "\t".join(
                (
                    dataset["label"],
                    "adaptive_oof",
                    fmt(metrics["counterfactual"]["acc"]),
                    fmt(metrics["counterfactual"]["delta"]),
                    fmt(metrics["commonsense"]["acc"]),
                    fmt(metrics["commonsense"]["delta"]),
                    str(dataset["combined_oof"]["delta_ci95"]["counterfactual"]),
                    str(dataset["combined_oof"]["delta_ci95"]["commonsense"]),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
