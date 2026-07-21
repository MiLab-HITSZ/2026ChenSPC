#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_hierarchical_eb_lambda_cv import resolved_baseline_key
from analyze_candidate_baseline_sweep import _method_block, _replay_pai
from analyze_candidate_baseline_frozen_transfer import _exact_mcnemar


VERSION = "qa_polarity_stress_analysis_v4"
VARIANTS = ("cs_first", "cf_first")
SIDES = ("counterfactual", "commonsense")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(paths: Mapping[str, str]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    output: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for split_variant, path in paths.items():
        split, variant = split_variant.split("_", 1)
        for source in read_jsonl(Path(path), "qa"):
            row = dict(source)
            key = (split_variant, str(row["pair_id"]), str(row["side"]))
            if key in output:
                raise ValueError(f"duplicate row: {key}")
            row["_split"] = split
            row["_variant"] = variant
            output[key] = row
    return output


def validate_protocol_rows(
    rows: Mapping[Tuple[str, str, str], Mapping[str, Any]]
) -> None:
    pair_sets = {}
    for split, expected_pairs in (("dev", 70), ("test", 230)):
        for variant in VARIANTS:
            key_prefix = f"{split}_{variant}"
            pair_ids = {
                key[1] for key in rows if key[0] == key_prefix
            }
            if len(pair_ids) != expected_pairs:
                raise ValueError(
                    f"{key_prefix} has {len(pair_ids)} pairs, expected {expected_pairs}"
                )
            incomplete = [
                pair_id
                for pair_id in pair_ids
                if {
                    key[2]
                    for key in rows
                    if key[0] == key_prefix and key[1] == pair_id
                }
                != set(SIDES)
            ]
            if incomplete:
                raise ValueError(f"incomplete sides in {key_prefix}: {incomplete[:5]}")
            pair_sets[(split, variant)] = pair_ids
        if pair_sets[(split, "cs_first")] != pair_sets[(split, "cf_first")]:
            raise ValueError(f"variant pair mismatch in {split}")

        for side in SIDES:
            labels = Counter(
                normalize_key(row.get("gt"))
                for key, row in rows.items()
                if key[0].startswith(f"{split}_") and key[2] == side
            )
            if labels != {"yes": expected_pairs, "no": expected_pairs}:
                raise ValueError(f"unbalanced {split}/{side} labels: {dict(labels)}")
    if pair_sets[("dev", "cs_first")] & pair_sets[("test", "cs_first")]:
        raise ValueError("development/test pair overlap")


def correct(prediction: str, row: Mapping[str, Any]) -> bool:
    return normalize_key(prediction) == normalize_key(row.get("gt"))


def side_metrics(
    rows: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    predictions: Mapping[Tuple[str, str, str], str],
    split_variant: str,
    side: str,
) -> Dict[str, Any]:
    selected = [
        (key, row)
        for key, row in rows.items()
        if key[0] == split_variant and key[2] == side
    ]
    baseline_predictions = [
        normalize_key(resolved_baseline_key(dict(row))) for _, row in selected
    ]
    method_predictions = [normalize_key(predictions[key]) for key, _ in selected]
    baseline_correct = sum(
        correct(prediction, row)
        for prediction, (_, row) in zip(baseline_predictions, selected)
    )
    method_correct = sum(
        correct(prediction, row)
        for prediction, (_, row) in zip(method_predictions, selected)
    )
    repairs = sum(
        not correct(resolved_baseline_key(dict(row)), row) and correct(predictions[key], row)
        for key, row in selected
    )
    harms = sum(
        correct(resolved_baseline_key(dict(row)), row) and not correct(predictions[key], row)
        for key, row in selected
    )
    n = len(selected)
    return {
        "n": n,
        "baseline_acc": baseline_correct / n,
        "method_acc": method_correct / n,
        "delta": (method_correct - baseline_correct) / n,
        "repairs": repairs,
        "harms": harms,
        "mcnemar_exact_p": _exact_mcnemar(repairs, harms),
        "baseline_yes_rate": sum(value == "yes" for value in baseline_predictions) / n,
        "method_yes_rate": sum(value == "yes" for value in method_predictions) / n,
        "answer_transitions": {
            f"{before or 'invalid'}_to_{after or 'invalid'}": sum(
                left == before and right == after
                for left, right in zip(baseline_predictions, method_predictions)
            )
            for before in ("yes", "no", "")
            for after in ("yes", "no", "")
            if any(
                left == before and right == after
                for left, right in zip(baseline_predictions, method_predictions)
            )
        },
    }


def predictions_from_analysis(
    rows: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    analysis_path: Path,
    dataset_names: Mapping[str, str],
) -> Dict[Tuple[str, str, str], str]:
    predictions = {
        key: resolved_baseline_key(dict(row))
        for key, row in rows.items()
        if key[0].startswith("test_")
    }
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    reverse_names = {value: key for key, value in dataset_names.items()}
    for case in payload["result"]["cases"]:
        dataset = str(case["dataset"])
        split_variant = reverse_names.get(dataset)
        if split_variant is None or not split_variant.startswith("test_"):
            continue
        key = (split_variant, str(case["pair_id"]), str(case["side"]))
        predictions[key] = normalize_key(case["prediction"])
    return predictions


def pai_predictions(
    rows: Mapping[Tuple[str, str, str], Mapping[str, Any]], gamma: float
) -> Dict[Tuple[str, str, str], str]:
    output = {}
    params = {
        "gamma": float(gamma),
        "beta": 0.1,
        "gate": "always",
        "contrast_margin": 0.0,
    }
    for key, row in rows.items():
        if not key[0].startswith("test_"):
            continue
        block = _method_block(dict(row), "pai")
        if block is None:
            raise ValueError(f"missing image/no-image candidate block at {key}")
        prediction = _replay_pai(block, params)
        if prediction is None:
            raise ValueError(f"PAI replay failed at {key}")
        output[key] = normalize_key(prediction)
    return output


def complement_metrics(
    rows: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    predictions: Mapping[Tuple[str, str, str], str],
) -> Dict[str, Any]:
    output = {}
    for side in SIDES:
        pair_ids = sorted(
            key[1]
            for key in rows
            if key[0] == "test_cs_first" and key[2] == side
        )
        complement = 0
        joint_correct = 0
        for pair_id in pair_ids:
            left_key = ("test_cs_first", pair_id, side)
            right_key = ("test_cf_first", pair_id, side)
            left = normalize_key(predictions[left_key])
            right = normalize_key(predictions[right_key])
            complement += int({left, right} == {"yes", "no"})
            joint_correct += int(
                correct(left, rows[left_key]) and correct(right, rows[right_key])
            )
        output[side] = {
            "n_pairs": len(pair_ids),
            "complement_consistency": complement / len(pair_ids),
            "joint_pair_accuracy": joint_correct / len(pair_ids),
        }
    return output


def score_view_diagnostics(
    rows: Mapping[Tuple[str, str, str], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Measure whether native, image-score, and prior views invert with wording."""

    def view_prediction(row: Mapping[str, Any], field: str) -> str:
        if field == "baseline_key":
            return normalize_key(resolved_baseline_key(dict(row)))
        block = row.get("cp_vbc")
        if not isinstance(block, dict):
            raise ValueError(f"missing candidate score block for {field}")
        prediction = normalize_key(block.get(field))
        if prediction not in {"yes", "no"}:
            raise ValueError(f"invalid {field}: {block.get(field)!r}")
        return prediction

    by_variant: Dict[str, Any] = {}
    for variant in VARIANTS:
        by_variant[variant] = {}
        for side in SIDES:
            selected = [
                row
                for key, row in rows.items()
                if key[0] == f"test_{variant}" and key[2] == side
            ]
            n = len(selected)
            metrics: Dict[str, Any] = {"n": n}
            for field, name in (
                ("baseline_key", "native"),
                ("image_top", "image_score"),
                ("prior_top", "no_image_prior"),
            ):
                predictions = [view_prediction(row, field) for row in selected]
                metrics[f"{name}_acc"] = sum(
                    correct(prediction, row)
                    for prediction, row in zip(predictions, selected)
                ) / n
                metrics[f"{name}_yes_rate"] = sum(
                    prediction == "yes" for prediction in predictions
                ) / n
            metrics["image_prior_disagreement_rate"] = sum(
                view_prediction(row, "image_top")
                != view_prediction(row, "prior_top")
                for row in selected
            ) / n
            by_variant[variant][side] = metrics

    paired_complement: Dict[str, Any] = {}
    for side in SIDES:
        pair_ids = sorted(
            key[1]
            for key in rows
            if key[0] == "test_cs_first" and key[2] == side
        )
        paired_complement[side] = {"n_pairs": len(pair_ids)}
        for field, name in (
            ("baseline_key", "native"),
            ("image_top", "image_score"),
            ("prior_top", "no_image_prior"),
        ):
            complementary = 0
            for pair_id in pair_ids:
                left = rows[("test_cs_first", pair_id, side)]
                right = rows[("test_cf_first", pair_id, side)]
                complementary += int(
                    {
                        view_prediction(left, field),
                        view_prediction(right, field),
                    }
                    == {"yes", "no"}
                )
            paired_complement[side][f"{name}_complement_consistency"] = (
                complementary / len(pair_ids)
            )

    return {
        "by_variant": by_variant,
        "paired_complement": paired_complement,
    }


def mapping_prediction(policy: str, baseline: str) -> str:
    baseline = normalize_key(baseline)
    if policy == "identity":
        return baseline
    if policy == "always_yes":
        return "yes"
    if policy == "always_no":
        return "no"
    if policy == "invert":
        return "no" if baseline == "yes" else "yes"
    raise ValueError(policy)


def evaluate_mapping(
    rows: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    split: str,
    policy: str,
) -> Dict[str, Any]:
    predictions = {
        key: mapping_prediction(policy, resolved_baseline_key(dict(row)))
        for key, row in rows.items()
        if key[0].startswith(f"{split}_")
    }
    by_variant = {
        variant: {
            side: side_metrics(rows, predictions, f"{split}_{variant}", side)
            for side in SIDES
        }
        for variant in VARIANTS
    }
    utility = sum(
        values["counterfactual"]["delta"]
        + 0.5 * values["commonsense"]["delta"]
        for values in by_variant.values()
    )
    eligible = all(
        values["counterfactual"]["delta"] >= -1e-12
        and values["commonsense"]["delta"] >= -0.2 - 1e-12
        for values in by_variant.values()
    )
    return {
        "policy": policy,
        "eligible": eligible,
        "utility_0_5_sum": utility,
        "by_variant": by_variant,
    }


def method_summary(
    rows: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    predictions: Mapping[Tuple[str, str, str], str],
) -> Dict[str, Any]:
    by_variant = {
        variant: {
            side: side_metrics(rows, predictions, f"test_{variant}", side)
            for side in SIDES
        }
        for variant in VARIANTS
    }
    macro = {
        side: {
            field: sum(by_variant[variant][side][field] for variant in VARIANTS)
            / len(VARIANTS)
            for field in ("baseline_acc", "method_acc", "delta")
        }
        for side in SIDES
    }
    baseline_predictions = {
        key: resolved_baseline_key(dict(row))
        for key, row in rows.items()
        if key[0].startswith("test_")
    }
    return {
        "by_variant": by_variant,
        "macro_balanced": macro,
        "baseline_paired_complement": complement_metrics(
            rows, baseline_predictions
        ),
        "paired_complement": complement_metrics(rows, predictions),
    }


def native_direction_strata(
    raw_path: Path, analysis_path: Path
) -> Dict[str, Any]:
    rows = read_jsonl(raw_path, "qa")
    predictions = {
        (str(row["pair_id"]), str(row["side"])): resolved_baseline_key(row)
        for row in rows
    }
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    for case in analysis["result"]["cases"]:
        if str(case.get("dataset")) == "test_qa":
            predictions[(str(case["pair_id"]), str(case["side"]))] = normalize_key(
                case["prediction"]
            )
    output = {}
    for side in SIDES:
        output[side] = {}
        for label in ("yes", "no"):
            selected = [
                row
                for row in rows
                if str(row["side"]) == side
                and normalize_key(row.get("gt")) == label
            ]
            baseline = [resolved_baseline_key(row) for row in selected]
            method = [
                predictions[(str(row["pair_id"]), str(row["side"]))]
                for row in selected
            ]
            n = len(selected)
            baseline_correct = sum(value == label for value in baseline)
            method_correct = sum(value == label for value in method)
            output[side][label] = {
                "n": n,
                "baseline_acc": baseline_correct / n if n else None,
                "method_acc": method_correct / n if n else None,
                "delta": (method_correct - baseline_correct) / n if n else None,
                "repairs": sum(
                    before != label and after == label
                    for before, after in zip(baseline, method)
                ),
                "harms": sum(
                    before == label and after != label
                    for before, after in zip(baseline, method)
                ),
            }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    for split in ("dev", "test"):
        for variant in VARIANTS:
            parser.add_argument(f"--{split}-{variant.replace('_', '-')}", required=True)
    parser.add_argument("--original-fit-analysis", required=True)
    parser.add_argument("--mc-only-fit-analysis", required=True)
    parser.add_argument("--balanced-fit-analysis", required=True)
    parser.add_argument("--native-raw", required=True)
    parser.add_argument("--native-analysis", required=True)
    parser.add_argument("--pai-gamma", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        f"{split}_{variant}": getattr(args, f"{split}_{variant}")
        for split in ("dev", "test")
        for variant in VARIANTS
    }
    rows = load_rows(paths)
    validate_protocol_rows(rows)
    dataset_names = {key: key for key in paths}
    original_path = Path(args.original_fit_analysis)
    mc_only_path = Path(args.mc_only_fit_analysis)
    balanced_path = Path(args.balanced_fit_analysis)
    native_raw_path = Path(args.native_raw)
    native_analysis_path = Path(args.native_analysis)
    original_predictions = predictions_from_analysis(rows, original_path, dataset_names)
    mc_only_predictions = predictions_from_analysis(rows, mc_only_path, dataset_names)
    balanced_predictions = predictions_from_analysis(rows, balanced_path, dataset_names)
    frozen_pai_predictions = pai_predictions(rows, args.pai_gamma)

    controls = [
        evaluate_mapping(rows, "dev", policy)
        for policy in ("identity", "always_yes", "always_no", "invert")
    ]
    selected = max(
        (row for row in controls if row["eligible"]),
        key=lambda row: (row["utility_0_5_sum"], row["policy"] == "identity"),
    )
    selected_test = evaluate_mapping(rows, "test", selected["policy"])

    payload = {
        "version": VERSION,
        "model": args.model,
        "inputs": {
            key: {"path": path, "sha256": sha256(Path(path))}
            for key, path in paths.items()
        },
        "protocol": {
            "labels_balanced_within_each_side": True,
            "paired_complementary_questions": True,
            "development_pairs": 70,
            "test_pairs": 230,
            "selection_uses_test_labels": False,
        },
        "native_benchmark_direction_strata": {
            "raw": str(native_raw_path),
            "raw_sha256": sha256(native_raw_path),
            "analysis": str(native_analysis_path),
            "analysis_sha256": sha256(native_analysis_path),
            "metrics": native_direction_strata(
                native_raw_path, native_analysis_path
            ),
        },
        "score_view_direction_diagnostics": score_view_diagnostics(rows),
        "original_fit": {
            "analysis": str(original_path),
            "analysis_sha256": sha256(original_path),
            **method_summary(rows, original_predictions),
        },
        "mc_only_fit": {
            "analysis": str(mc_only_path),
            "analysis_sha256": sha256(mc_only_path),
            **method_summary(rows, mc_only_predictions),
        },
        "balanced_fit": {
            "analysis": str(balanced_path),
            "analysis_sha256": sha256(balanced_path),
            **method_summary(rows, balanced_predictions),
        },
        "frozen_pai": {
            "gamma": float(args.pai_gamma),
            "beta": 0.1,
            "selection_source": "native 70-pair development split",
            **method_summary(rows, frozen_pai_predictions),
        },
        "direction_only_controls": {
            "development": controls,
            "selected_policy": selected["policy"],
            "test": selected_test,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("fit\tvariant\tdCF\tdCS\tCF_joint\tCS_joint")
    for fit in ("original_fit", "mc_only_fit", "balanced_fit"):
        summary = payload[fit]
        for variant in VARIANTS:
            values = summary["by_variant"][variant]
            print(
                f"{fit}\t{variant}\t{values['counterfactual']['delta']:+.4f}"
                f"\t{values['commonsense']['delta']:+.4f}"
                f"\t{summary['paired_complement']['counterfactual']['joint_pair_accuracy']:.4f}"
                f"\t{summary['paired_complement']['commonsense']['joint_pair_accuracy']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
