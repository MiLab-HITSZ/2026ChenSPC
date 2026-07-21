import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_hierarchical_eb_lambda_cv import (
    exact_mcnemar_pvalue,
    resolved_baseline_key,
)


SIDES = ("counterfactual", "commonsense")


def row_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("task") or row.get("_task") or ""),
        str(row.get("pair_id") or ""),
        str(row.get("side") or ""),
    )


def candidate_scores(row: Mapping[str, Any]) -> Tuple[List[str], List[float], List[float]]:
    candidates = (row.get("cp_vbc") or {}).get("candidates") or []
    keys = [normalize_key(candidate.get("key")) for candidate in candidates]
    image = [float(candidate["logp_image"]) for candidate in candidates]
    prior = [float(candidate["logp_prior"]) for candidate in candidates]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(f"invalid candidates for {row_key(row)}")
    return keys, image, prior


def top_key(keys: Sequence[str], scores: Sequence[float]) -> str:
    return keys[max(range(len(keys)), key=lambda index: scores[index])]


def fixed_lambda_prediction(row: Mapping[str, Any], lambda_value: float) -> str:
    keys, image, prior = candidate_scores(row)
    corrected = [
        image_score - float(lambda_value) * prior_score
        for image_score, prior_score in zip(image, prior)
    ]
    return top_key(keys, corrected)


def bootstrap_delta(
    baseline_correct: Sequence[int],
    method_correct: Sequence[int],
    samples: int,
    seed: int,
) -> List[float]:
    differences = [method - baseline for baseline, method in zip(baseline_correct, method_correct)]
    if not differences or samples <= 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    draws = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(max(1, int(samples)))
    )
    return [
        float(draws[int(0.025 * (len(draws) - 1))]),
        float(draws[int(0.975 * (len(draws) - 1))]),
    ]


def side_summary(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[Tuple[str, str, str], str],
    side: str,
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    selected = [row for row in rows if row.get("side") == side]
    baseline_correct = [
        int(resolved_baseline_key(dict(row)) == normalize_key(row.get("gt")))
        for row in selected
    ]
    method_correct = [
        int(predictions[row_key(row)] == normalize_key(row.get("gt")))
        for row in selected
    ]
    repairs = sum(not baseline and method for baseline, method in zip(baseline_correct, method_correct))
    harms = sum(baseline and not method for baseline, method in zip(baseline_correct, method_correct))
    overrides = sum(
        predictions[row_key(row)] != resolved_baseline_key(dict(row)) for row in selected
    )
    count = len(selected)
    baseline_acc = sum(baseline_correct) / count if count else 0.0
    method_acc = sum(method_correct) / count if count else 0.0
    return {
        "n": count,
        "baseline_acc": baseline_acc,
        "acc": method_acc,
        "delta": method_acc - baseline_acc,
        "delta_ci95": bootstrap_delta(
            baseline_correct, method_correct, bootstrap, seed
        ),
        "overrides": overrides,
        "repairs": repairs,
        "harms": harms,
        "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(repairs, harms),
    }


def policy_summary(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[Tuple[str, str, str], str],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    by_side = {
        side: side_summary(rows, predictions, side, bootstrap, seed + index)
        for index, side in enumerate(SIDES)
    }
    cf_acc = by_side["counterfactual"]["acc"]
    cs_acc = by_side["commonsense"]["acc"]
    paired_cs_gt = {
        (str(row.get("task")), str(row.get("pair_id"))): normalize_key(row.get("gt"))
        for row in rows
        if row.get("side") == "commonsense"
    }
    cf_errors = [
        row
        for row in rows
        if row.get("side") == "counterfactual"
        and predictions[row_key(row)] != normalize_key(row.get("gt"))
    ]
    prior_attraction = sum(
        predictions[row_key(row)]
        == paired_cs_gt.get((str(row.get("task")), str(row.get("pair_id"))))
        for row in cf_errors
    )
    return {
        "by_side": by_side,
        "cfad": cs_acc - cf_acc,
        "rpd": (cs_acc - cf_acc) / cs_acc if cs_acc else None,
        "cf_error_count": len(cf_errors),
        "prior_attraction_among_cf_errors": (
            prior_attraction / len(cf_errors) if cf_errors else None
        ),
    }


def reconstruct_heb_predictions(
    rows: Sequence[Mapping[str, Any]], heb_path: Path
) -> Dict[Tuple[str, str, str], str]:
    predictions = {
        row_key(row): resolved_baseline_key(dict(row)) for row in rows
    }
    artifact = json.loads(heb_path.read_text(encoding="utf-8"))
    for case in artifact["result"]["cases"]:
        key = (
            str(case.get("task") or ""),
            str(case.get("pair_id") or ""),
            str(case.get("side") or ""),
        )
        if key in predictions:
            predictions[key] = normalize_key(case.get("prediction"))
    return predictions


def prior_alignment(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cs_gt = {
        (str(row.get("task")), str(row.get("pair_id"))): normalize_key(row.get("gt"))
        for row in rows
        if row.get("side") == "commonsense"
    }
    output: Dict[str, Any] = {}
    for task in sorted({str(row.get("task")) for row in rows}):
        cf_rows = [
            row
            for row in rows
            if row.get("task") == task and row.get("side") == "counterfactual"
        ]
        records = []
        for row in cf_rows:
            keys, image, prior = candidate_scores(row)
            baseline = resolved_baseline_key(dict(row))
            prior_top = top_key(keys, prior)
            image_top = top_key(keys, image)
            normal_gt = cs_gt[(task, str(row.get("pair_id")))]
            cf_gt = normalize_key(row.get("gt"))
            records.append(
                {
                    "baseline_error": baseline != cf_gt,
                    "baseline_equals_prior_top": baseline == prior_top,
                    "baseline_equals_cs_gt": baseline == normal_gt,
                    "prior_top_equals_cs_gt": prior_top == normal_gt,
                    "prior_top_equals_cf_gt": prior_top == cf_gt,
                    "image_top_equals_cf_gt": image_top == cf_gt,
                }
            )
        errors = [record for record in records if record["baseline_error"]]

        def rates(selected: Sequence[Mapping[str, bool]]) -> Dict[str, Any]:
            fields = [key for key in records[0] if key != "baseline_error"] if records else []
            return {
                "n": len(selected),
                **{
                    field: sum(bool(record[field]) for record in selected) / len(selected)
                    if selected
                    else None
                    for field in fields
                },
            }

        output[task] = {
            "all_counterfactual": rates(records),
            "native_counterfactual_errors": rates(errors),
        }
    return output


def build_predictions(
    rows: Iterable[Mapping[str, Any]], lambda_value: float
) -> Dict[Tuple[str, str, str], str]:
    return {
        row_key(row): fixed_lambda_prediction(row, lambda_value) for row in rows
    }


def analyze_model(
    results_path: Path,
    heb_path: Path,
    lambdas: Sequence[float],
    bootstrap: int,
    sweep_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    rows = read_jsonl(results_path, "qa") + read_jsonl(results_path, "mc")
    rows.sort(key=row_key)
    by_task = {
        task: [row for row in rows if row.get("task") == task]
        for task in ("mc", "qa")
    }
    native = {
        row_key(row): resolved_baseline_key(row) for row in rows
    }
    heb = reconstruct_heb_predictions(rows, heb_path)
    policies: Dict[str, Dict[str, Any]] = {}
    for task_index, (task, task_rows) in enumerate(by_task.items()):
        task_native = {row_key(row): native[row_key(row)] for row in task_rows}
        task_heb = {row_key(row): heb[row_key(row)] for row in task_rows}
        policies[task] = {
            "native_generation": policy_summary(
                task_rows, task_native, bootstrap, seed + task_index * 100
            ),
            "image_candidate_lambda_0": policy_summary(
                task_rows,
                build_predictions(task_rows, 0.0),
                bootstrap,
                seed + task_index * 100 + 10,
            ),
            "h_eb_spc": policy_summary(
                task_rows, task_heb, bootstrap, seed + task_index * 100 + 20
            ),
            "lambda_sweep": {
                f"{value:.3f}": policy_summary(
                    task_rows,
                    build_predictions(task_rows, value),
                    sweep_bootstrap,
                    seed + task_index * 100 + 30 + lambda_index,
                )
                for lambda_index, value in enumerate(lambdas)
            },
        }
    return {
        "results": str(results_path),
        "h_eb": str(heb_path),
        "rows": len(rows),
        "prior_alignment": prior_alignment(rows),
        "policies": policies,
    }


def parse_model(value: str) -> Tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or ":" not in parts[1]:
        raise argparse.ArgumentTypeError(
            "--model must be LABEL=RESULTS_JSONL:H_EB_JSON"
        )
    label = parts[0].strip()
    results, heb = parts[1].rsplit(":", 1)
    if not label:
        raise argparse.ArgumentTypeError("model label cannot be empty")
    return label, Path(results).expanduser(), Path(heb).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate candidate scoring from prior-residual correction."
    )
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda-start", type=float, default=0.0)
    parser.add_argument("--lambda-stop", type=float, default=1.5)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument(
        "--sweep-bootstrap",
        type=int,
        default=0,
        help="Bootstrap draws for each diagnostic lambda point; zero skips resampling.",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    if args.lambda_step <= 0 or args.lambda_stop < args.lambda_start:
        raise SystemExit("invalid lambda range")
    count = int(round((args.lambda_stop - args.lambda_start) / args.lambda_step))
    lambdas = [args.lambda_start + index * args.lambda_step for index in range(count + 1)]
    output = {
        "version": "spc_candidate_attribution_v1",
        "interpretation": (
            "The no-image candidate distribution is the language/world-prior "
            "estimator established by REVIS; this analysis attributes gains "
            "between native generation, image-only candidate scoring, and prior subtraction."
        ),
        "models": {
            label: analyze_model(
                results,
                heb,
                lambdas,
                args.bootstrap,
                args.sweep_bootstrap,
                args.seed + index * 1000,
            )
            for index, (label, results, heb) in enumerate(args.model)
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("model\ttask\tpolicy\tCF\tdCF\tCS\tdCS\tCF_repairs/harms\tCS_repairs/harms")
    for label, model in output["models"].items():
        for task, policies in model["policies"].items():
            for policy in ("native_generation", "image_candidate_lambda_0", "h_eb_spc"):
                summary = policies[policy]["by_side"]
                cf = summary["counterfactual"]
                cs = summary["commonsense"]
                print(
                    f"{label}\t{task}\t{policy}\t{cf['acc']:.3f}\t{cf['delta']:+.3f}"
                    f"\t{cs['acc']:.3f}\t{cs['delta']:+.3f}"
                    f"\t{cf['repairs']}/{cf['harms']}\t{cs['repairs']}/{cs['harms']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
