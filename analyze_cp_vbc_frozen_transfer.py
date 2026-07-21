import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analyze_cp_vbc_bayes_path_cv import (
    SIDES,
    bayes_path_features,
    bayes_path_grid,
    fixed_features,
    fixed_grid,
    metric,
    metric_by_field,
    normalize_key,
    predict_bayes_path,
    predict_fixed,
    read_jsonl,
    select_params,
)
from analyze_hierarchical_eb_lambda_cv import exact_mcnemar_pvalue


VERSION = "cp_vbc_frozen_transfer_v1"


def resolved_baseline_key(row: Dict[str, Any]) -> str:
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
        return ""
    return normalize_key(cp_vbc.get("baseline_key"))


def predictions_for_params(
    rows: Sequence[Dict[str, Any]], params: Dict[str, Any]
) -> List[Optional[str]]:
    if params["family"] == "fixed":
        return [
            predict_fixed(fixed_features(row, float(params["lambda"])), params)
            for row in rows
        ]
    return [
        predict_bayes_path(
            bayes_path_features(
                row,
                float(params["lambda_low"]),
                float(params["lambda_high"]),
                int(params["lambda_steps"]),
            ),
            params,
        )
        for row in rows
    ]


def bootstrap_subset_delta(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Optional[str]],
    side: str,
    samples: int,
    seed: int,
) -> List[float]:
    values = [
        int(normalize_key(prediction) == normalize_key(row.get("gt")))
        - int(
            resolved_baseline_key(row)
            == normalize_key(row.get("gt"))
        )
        for row, prediction in zip(rows, predictions)
        if row.get("side") == side
    ]
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    draws = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(max(1, int(samples)))
    )
    return [
        float(draws[int(0.025 * (len(draws) - 1))]),
        float(draws[int(0.975 * (len(draws) - 1))]),
    ]


def paired_test(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Optional[str]],
) -> Dict[str, Dict[str, Any]]:
    output = {}
    for side in SIDES:
        repairs = 0
        harms = 0
        for row, prediction in zip(rows, predictions):
            if row.get("side") != side:
                continue
            gt = normalize_key(row.get("gt"))
            baseline_correct = resolved_baseline_key(row) == gt
            prediction_correct = normalize_key(prediction) == gt
            repairs += int(not baseline_correct and prediction_correct)
            harms += int(baseline_correct and not prediction_correct)
        output[side] = {
            "repairs": repairs,
            "harms": harms,
            "discordant": repairs + harms,
            "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(repairs, harms),
        }
    return output


def development_prediction_banks(
    rows: Sequence[Dict[str, Any]], path_steps: int
) -> Tuple[List[Dict[str, Any]], List[List[Optional[str]]]]:
    fixed_params = fixed_grid()
    fixed_cache = {
        value: [fixed_features(row, value) for row in rows]
        for value in sorted({float(params["lambda"]) for params in fixed_params})
    }
    fixed_predictions = [
        [
            predict_fixed(features, params)
            for features in fixed_cache[float(params["lambda"])]
        ]
        for params in fixed_params
    ]

    path_params = bayes_path_grid(path_steps)
    path_cache = {}
    for params in path_params:
        interval = (float(params["lambda_low"]), float(params["lambda_high"]))
        if interval not in path_cache:
            path_cache[interval] = [
                bayes_path_features(row, interval[0], interval[1], path_steps)
                for row in rows
            ]
    path_predictions = [
        [
            predict_bayes_path(
                features,
                params,
            )
            for features in path_cache[
                (float(params["lambda_low"]), float(params["lambda_high"]))
            ]
        ]
        for params in path_params
    ]
    return fixed_params + path_params, fixed_predictions + path_predictions


def summarize_selected(
    dev_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    params: Dict[str, Any],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    dev_predictions = predictions_for_params(dev_rows, params)
    test_predictions = predictions_for_params(test_rows, params)
    return {
        "params": params,
        "development_metrics": metric(dev_rows, dev_predictions, range(len(dev_rows))),
        "test_metrics": metric(test_rows, test_predictions, range(len(test_rows))),
        "test_by_subcategory": metric_by_field(
            test_rows, test_predictions, range(len(test_rows)), "subcategory"
        ),
        "test_delta_ci95": {
            side: bootstrap_subset_delta(
                test_rows,
                test_predictions,
                side,
                bootstrap,
                seed + side_index,
            )
            for side_index, side in enumerate(SIDES)
        },
        "test_paired_tests": paired_test(test_rows, test_predictions),
    }


def run_frozen_transfer(
    development_path: Path,
    test_path: Path,
    task: str,
    max_cs_drop: float,
    selection_objective: str,
    cs_cost: float,
    path_steps: int,
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    development = read_jsonl(development_path, task)
    test = read_jsonl(test_path, task)
    if not development or not test:
        raise ValueError("development and test inputs must both contain usable rows")
    development_pairs = {str(row.get("pair_id")) for row in development}
    test_pairs = {str(row.get("pair_id")) for row in test}
    overlap = development_pairs & test_pairs
    if overlap:
        raise ValueError(f"development/test pair overlap: {sorted(overlap)[:10]}")

    params, prediction_bank = development_prediction_banks(development, path_steps)
    dev_ids = list(range(len(development)))
    selected = {}
    family_indices = {
        family: [index for index, value in enumerate(params) if value["family"] == family]
        for family in ("fixed", "bayes_path")
    }
    for family, indices in family_indices.items():
        local_params = [params[index] for index in indices]
        local_bank = [prediction_bank[index] for index in indices]
        local_index = select_params(
            development,
            local_bank,
            local_params,
            dev_ids,
            max_cs_drop,
            selection_objective=selection_objective,
            cs_cost=cs_cost,
        )
        selected[family] = summarize_selected(
            development,
            test,
            local_params[local_index],
            bootstrap,
            seed + len(selected) * 10,
        )

    adaptive_index = select_params(
        development,
        prediction_bank,
        params,
        dev_ids,
        max_cs_drop,
        selection_objective=selection_objective,
        cs_cost=cs_cost,
    )
    selected["adaptive"] = summarize_selected(
        development,
        test,
        params[adaptive_index],
        bootstrap,
        seed + 100,
    )
    return {
        "version": VERSION,
        "task": task,
        "development_path": str(development_path),
        "test_path": str(test_path),
        "development_pairs": len(development_pairs),
        "test_pairs": len(test_pairs),
        "selection": {
            "max_cs_drop": float(max_cs_drop),
            "objective": selection_objective,
            "cs_cost": float(cs_cost),
            "path_steps": int(path_steps),
            "target_labels_used_for_selection": False,
        },
        "results": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit fixed/BayesPath CPRC on development rows and apply once to a disjoint test set."
    )
    parser.add_argument("--development", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--task", choices=("qa", "mc"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cs-drop", type=float, default=0.2)
    parser.add_argument(
        "--selection-objective",
        choices=("cf_first", "net_utility"),
        default="net_utility",
    )
    parser.add_argument("--cs-cost", type=float, default=0.5)
    parser.add_argument("--path-steps", type=int, default=41)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    result = run_frozen_transfer(
        Path(args.development),
        Path(args.test),
        args.task,
        args.max_cs_drop,
        args.selection_objective,
        max(0.0, float(args.cs_cost)),
        max(2, int(args.path_steps)),
        max(1, int(args.bootstrap)),
        int(args.seed),
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("method\tCF_Acc\tdCF\tCS_Acc\tdCS\tCF_CI95\tCS_CI95")
    for method, values in result["results"].items():
        metrics = values["test_metrics"]
        print(
            "\t".join(
                (
                    method,
                    f"{metrics['counterfactual']['acc']:.3f}",
                    f"{metrics['counterfactual']['delta']:+.3f}",
                    f"{metrics['commonsense']['acc']:.3f}",
                    f"{metrics['commonsense']['delta']:+.3f}",
                    str(values["test_delta_ci95"]["counterfactual"]),
                    str(values["test_delta_ci95"]["commonsense"]),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
