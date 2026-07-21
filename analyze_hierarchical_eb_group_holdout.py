import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from analyze_cp_vbc_bayes_path_cv import metric, normalize_key, read_jsonl
from analyze_hierarchical_eb_lambda_cv import (
    context_features,
    exact_mcnemar_pvalue,
    fit_and_select,
    resolved_baseline_key,
)


def bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    ids: Sequence[int],
    side: str,
    samples: int,
    seed: int,
) -> List[float]:
    values = [
        int(normalize_key(predictions[index]) == normalize_key(rows[index].get("gt")))
        - int(resolved_baseline_key(dict(rows[index])) == normalize_key(rows[index].get("gt")))
        for index in ids
        if rows[index].get("side") == side
    ]
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    draws = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(max(1, samples))
    )
    return [
        float(draws[int(0.025 * (len(draws) - 1))]),
        float(draws[int(0.975 * (len(draws) - 1))]),
    ]


def paired_counts(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[str], ids: Sequence[int], side: str
) -> Dict[str, Any]:
    repairs = 0
    harms = 0
    overrides = 0
    for index in ids:
        row = rows[index]
        if row.get("side") != side:
            continue
        gt = normalize_key(row.get("gt"))
        baseline = resolved_baseline_key(dict(row))
        prediction = normalize_key(predictions[index])
        overrides += prediction != baseline
        repairs += baseline != gt and prediction == gt
        harms += baseline == gt and prediction != gt
    return {
        "overrides": overrides,
        "repairs": repairs,
        "harms": harms,
        "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(repairs, harms),
    }


def load_rows(
    development: Sequence[Path], test: Sequence[Path]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for split, paths in (("development", development), ("test", test)):
        for path in paths:
            for task in ("mc", "qa"):
                for source in read_jsonl(path, task):
                    identity = (
                        split,
                        task,
                        str(source.get("pair_id")),
                        str(source.get("side")),
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    row = dict(source)
                    row["_split"] = split
                    row["_task"] = task
                    row["_dataset"] = f"{split}_{task}"
                    rows.append(row)
    return rows


def aggregate(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[str], test_ids: Sequence[int]
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for task in ("mc", "qa"):
        task_ids = [index for index in test_ids if rows[index].get("_task") == task]
        output[task] = metric(rows, predictions, task_ids)
    return output


def run_group_holdout(
    rows: Sequence[Dict[str, Any]],
    group_field: str,
    max_cs_drop: float,
    quadrature_points: int,
    bootstrap: int,
    seed: int,
    selection_objective: str,
    cs_cost: float,
) -> Dict[str, Any]:
    contexts = np.stack([context_features(row) for row in rows])
    baseline = [resolved_baseline_key(row) for row in rows]
    groups = sorted(
        {
            str(row.get(group_field))
            for row in rows
            if row.get("_split") == "test" and row.get(group_field) is not None
        }
    )
    predictions = list(baseline)
    folds = []
    heldout_ids: List[int] = []
    for group_index, group in enumerate(groups):
        train_ids = [
            index
            for index, row in enumerate(rows)
            if row.get("_split") == "development"
            and str(row.get(group_field)) != group
        ]
        test_ids = [
            index
            for index, row in enumerate(rows)
            if row.get("_split") == "test" and str(row.get(group_field)) == group
        ]
        fold_predictions, _, params = fit_and_select(
            rows,
            contexts,
            train_ids,
            max_cs_drop,
            True,
            quadrature_points,
            selection_objective,
            cs_cost,
        )
        for index in test_ids:
            predictions[index] = fold_predictions[index]
        heldout_ids.extend(test_ids)
        fold_metrics = aggregate(rows, fold_predictions, test_ids)
        fold_stats = {
            task: {
                side: paired_counts(
                    rows,
                    fold_predictions,
                    [index for index in test_ids if rows[index].get("_task") == task],
                    side,
                )
                for side in ("counterfactual", "commonsense")
            }
            for task in ("mc", "qa")
        }
        folds.append(
            {
                "heldout_group": group,
                "train_rows": len(train_ids),
                "test_rows": len(test_ids),
                "train_groups": sorted(
                    {
                        str(rows[index].get(group_field)) for index in train_ids
                    }
                ),
                "metrics": fold_metrics,
                "paired_counts": fold_stats,
                "selected": params,
            }
        )

    overall = aggregate(rows, predictions, heldout_ids)
    uncertainty = {}
    paired = {}
    for task_index, task in enumerate(("mc", "qa")):
        task_ids = [index for index in heldout_ids if rows[index].get("_task") == task]
        uncertainty[task] = {
            side: bootstrap_delta(
                rows,
                predictions,
                task_ids,
                side,
                bootstrap,
                seed + task_index * 10 + side_index,
            )
            for side_index, side in enumerate(("counterfactual", "commonsense"))
        }
        paired[task] = {
            side: paired_counts(rows, predictions, task_ids, side)
            for side in ("counterfactual", "commonsense")
        }
    return {
        "group_field": group_field,
        "groups": groups,
        "folds": folds,
        "overall": overall,
        "delta_ci95": uncertainty,
        "paired_counts": paired,
        "heldout_rows": len(heldout_ids),
        "guarantee": (
            "Every evaluated row belongs to a group absent from calibration; "
            "group labels are used only to construct folds and never as inference features."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate H-EB with entire CDH groups absent from calibration."
    )
    parser.add_argument("--development", action="append", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--group-field", choices=("subcategory", "category"), default="subcategory"
    )
    parser.add_argument("--max-cs-drop", type=float, default=0.10)
    parser.add_argument("--quadrature-points", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--selection-objective", choices=("cf_first", "net_utility"), default="net_utility"
    )
    parser.add_argument("--cs-cost", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_rows(
        [Path(path) for path in args.development],
        [Path(path) for path in args.test],
    )
    result = run_group_holdout(
        rows,
        args.group_field,
        abs(float(args.max_cs_drop)),
        max(8, int(args.quadrature_points)),
        max(1, int(args.bootstrap)),
        int(args.seed),
        args.selection_objective,
        max(0.0, float(args.cs_cost)),
    )
    output = {
        "version": "hierarchical_eb_group_holdout_v1",
        "development": args.development,
        "test": args.test,
        "max_cs_drop": abs(float(args.max_cs_drop)),
        "selection_objective": args.selection_objective,
        "cs_cost": max(0.0, float(args.cs_cost)),
        "result": result,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("task\tCF\tdCF\tCS\tdCS\tCF_CI95\tCS_CI95")
    for task, values in result["overall"].items():
        cf = values["counterfactual"]
        cs = values["commonsense"]
        print(
            f"{task}\t{cf['acc']:.3f}\t{cf['delta']:+.3f}\t{cs['acc']:.3f}"
            f"\t{cs['delta']:+.3f}\t{result['delta_ci95'][task]['counterfactual']}"
            f"\t{result['delta_ci95'][task]['commonsense']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
