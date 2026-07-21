import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_hierarchical_eb_lambda_cv import resolved_baseline_key, run_cv


def exact_mcnemar(repairs: int, harms: int) -> float:
    discordant = repairs + harms
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(repairs, harms) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def multinomial_bootstrap_ci(
    repairs: int, harms: int, n: int, draws: int, seed: int
) -> List[float]:
    import numpy as np

    if n <= 0:
        return [0.0, 0.0]
    if draws <= 0:
        delta = (repairs - harms) / n
        return [delta, delta]
    probabilities = [repairs / n, harms / n, (n - repairs - harms) / n]
    samples = np.random.default_rng(seed).multinomial(n, probabilities, size=draws)
    differences = (samples[:, 0] - samples[:, 1]) / n
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def load_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    output = []
    seen = set()
    for path in paths:
        for row in read_jsonl(path, "qa"):
            key = (str(row.get("protocol")), int(row.get("question_id")))
            if key in seen:
                raise ValueError(f"duplicate POPE row: {key}")
            seen.add(key)
            copied = dict(row)
            copied["_dataset"] = f"pope_{key[0]}"
            copied["_task"] = "qa"
            output.append(copied)
    return output


def binary_metrics(labels: Sequence[str], predictions: Sequence[str]) -> Dict[str, Any]:
    tp = sum(label == "yes" and pred == "yes" for label, pred in zip(labels, predictions))
    tn = sum(label == "no" and pred == "no" for label, pred in zip(labels, predictions))
    fp = sum(label == "no" and pred == "yes" for label, pred in zip(labels, predictions))
    fn = sum(label == "yes" and pred == "no" for label, pred in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(labels),
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_rate": sum(pred == "yes" for pred in predictions) / len(predictions)
        if predictions
        else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def reconstruct_predictions(
    rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any]
) -> Dict[Tuple[str, str, str], str]:
    predictions = {
        (str(row["_dataset"]), str(row["pair_id"]), str(row["side"])): resolved_baseline_key(dict(row))
        for row in rows
        if str(row["_dataset"]).startswith("pope_")
    }
    for case in result["cases"]:
        dataset = str(case["dataset"])
        if dataset.startswith("pope_"):
            predictions[(dataset, str(case["pair_id"]), str(case["side"]))] = normalize_key(
                case["prediction"]
            )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply frozen CDH CPRC to native POPE.")
    parser.add_argument("--pope", action="append", required=True)
    parser.add_argument("--dev-mc", required=True)
    parser.add_argument("--dev-qa", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for label, task, path in (
        ("dev_mc", "mc", args.dev_mc),
        ("dev_qa", "qa", args.dev_qa),
    ):
        for source in read_jsonl(Path(path), task):
            row = dict(source)
            row["_dataset"] = label
            row["_task"] = task
            rows.append(row)
    pope_rows = load_rows(Path(path) for path in args.pope)
    rows.extend(pope_rows)

    # POPE rows are independent binary questions. We compute their paired
    # bootstrap below from the equivalent repair/harm multinomial, avoiding the
    # generic Python resampling loop over 9,000 rows.
    result = run_cv(
        rows,
        "frozen_transfer",
        5,
        0.20,
        True,
        64,
        1,
        int(args.seed),
        "net_utility",
        0.5,
        ("dev_mc", "dev_qa"),
    )
    cprc = reconstruct_predictions(rows, result)
    native_results = {}
    for protocol_index, protocol in enumerate(("random", "popular", "adversarial")):
        dataset = f"pope_{protocol}"
        selected = sorted(
            (row for row in pope_rows if row["_dataset"] == dataset),
            key=lambda row: int(row["question_id"]),
        )
        if not selected:
            continue
        labels = [normalize_key(row["gt"]) for row in selected]
        baseline = [resolved_baseline_key(row) for row in selected]
        corrected = [
            cprc[(dataset, str(row["pair_id"]), str(row["side"]))]
            for row in selected
        ]
        baseline_metrics = binary_metrics(labels, baseline)
        cprc_metrics = binary_metrics(labels, corrected)
        repairs = sum(
            base != label and fixed == label
            for label, base, fixed in zip(labels, baseline, corrected)
        )
        harms = sum(
            base == label and fixed != label
            for label, base, fixed in zip(labels, baseline, corrected)
        )
        native_results[protocol] = {
            "baseline": baseline_metrics,
            "cprc": cprc_metrics,
            "delta_accuracy": cprc_metrics["accuracy"] - baseline_metrics["accuracy"],
            "delta_f1": cprc_metrics["f1"] - baseline_metrics["f1"],
            "interventions": sum(a != b for a, b in zip(baseline, corrected)),
            "intervention_rate": sum(a != b for a, b in zip(baseline, corrected))
            / len(selected),
            "paired_test": {
                "repairs": repairs,
                "harms": harms,
                "discordant": repairs + harms,
                "exact_mcnemar_p_two_sided": exact_mcnemar(repairs, harms),
            },
            "delta_accuracy_ci95": multinomial_bootstrap_ci(
                repairs,
                harms,
                len(selected),
                int(args.bootstrap),
                int(args.seed) + protocol_index + 1,
            ),
        }

    selected = sorted(
        pope_rows,
        key=lambda row: (str(row["protocol"]), int(row["question_id"])),
    )
    labels = [normalize_key(row["gt"]) for row in selected]
    baseline = [resolved_baseline_key(row) for row in selected]
    corrected = [
        cprc[
            (
                str(row["_dataset"]),
                str(row["pair_id"]),
                str(row["side"]),
            )
        ]
        for row in selected
    ]
    baseline_metrics = binary_metrics(labels, baseline)
    cprc_metrics = binary_metrics(labels, corrected)
    repairs = sum(
        base != label and fixed == label
        for label, base, fixed in zip(labels, baseline, corrected)
    )
    harms = sum(
        base == label and fixed != label
        for label, base, fixed in zip(labels, baseline, corrected)
    )
    interventions = sum(base != fixed for base, fixed in zip(baseline, corrected))
    native_results["overall"] = {
        "baseline": baseline_metrics,
        "cprc": cprc_metrics,
        "delta_accuracy": cprc_metrics["accuracy"] - baseline_metrics["accuracy"],
        "delta_f1": cprc_metrics["f1"] - baseline_metrics["f1"],
        "interventions": interventions,
        "intervention_rate": interventions / len(selected),
        "paired_test": {
            "repairs": repairs,
            "harms": harms,
            "discordant": repairs + harms,
            "exact_mcnemar_p_two_sided": exact_mcnemar(repairs, harms),
        },
        "delta_accuracy_ci95": multinomial_bootstrap_ci(
            repairs, harms, len(selected), int(args.bootstrap), int(args.seed)
        ),
    }

    artifact = {
        "version": "pope_native_cprc_v1",
        "protocol": "All official COCO random/popular/adversarial questions; native yes/no metrics.",
        "calibration": "Frozen 70-pair CDH QA+MC development fit; no POPE labels used.",
        "verified_rows": len(pope_rows),
        "raw_sources": args.pope,
        "results": native_results,
        "fit_selection": result["selections"],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(native_results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
