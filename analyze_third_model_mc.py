#!/usr/bin/env python3
"""Core third-model audit: MC mechanism alignment, repair, and family transfer."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_cprc_paired_calibration import (
    VariantSelector,
    endpoint_metrics,
    fit_regime,
    selected_ids,
)
from analyze_hierarchical_eb_lambda_cv import resolved_baseline_key


VERSION = "cprc_third_model_mc_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for label, path in (("dev_mc", spec["dev_mc"]), ("test_mc", spec["test_mc"])):
        for source in read_jsonl(Path(path), "mc"):
            row = copy.deepcopy(source)
            key = (str(row.get("pair_id")), str(row.get("side")))
            if key in seen:
                raise ValueError(f"duplicate MC row: {key}")
            seen.add(key)
            row["_dataset"] = label
            row["_task"] = "mc"
            rows.append(row)
    return rows


def alignment_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    test_cf = [
        row
        for row in rows
        if str(row.get("_dataset")) == "test_mc"
        and str(row.get("side")) == "counterfactual"
    ]
    errors = [
        row
        for row in test_cf
        if normalize_key(resolved_baseline_key(dict(row)))
        != normalize_key(row.get("gt"))
    ]
    baseline_prior = sum(
        normalize_key(resolved_baseline_key(dict(row)))
        == normalize_key((row.get("cp_vbc") or {}).get("prior_top"))
        for row in errors
    )
    prior_cs = sum(
        normalize_key((row.get("cp_vbc") or {}).get("prior_top"))
        == normalize_key(row.get("cs_gt"))
        for row in errors
    )
    return {
        "test_cf_rows": len(test_cf),
        "native_cf_errors": len(errors),
        "native_error_equals_prior_top": baseline_prior,
        "native_error_equals_prior_top_rate": baseline_prior / len(errors) if errors else None,
        "prior_top_equals_paired_cs_answer": prior_cs,
        "prior_top_equals_paired_cs_answer_rate": prior_cs / len(errors) if errors else None,
    }


def selector(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    selection_ids: Sequence[int],
) -> VariantSelector:
    return VariantSelector(
        rows,
        "paired_calibration",
        float(config["paired_cs_budget"]),
        float(config["paired_cs_cost"]),
        bool(config["require_nonnegative_cf_per_task"]),
        tasks=("mc",),
        selection_ids=selection_ids,
    )


def compact(result: Dict[str, Any]) -> Dict[str, Any]:
    result.pop("test_predictions", None)
    return result


def run_model(model: str, spec: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    rows = load_rows(spec)
    dev_ids = selected_ids(rows, "development")
    test_ids = selected_ids(rows, "test")
    full_selector = selector(rows, config, dev_ids)
    full_fit = fit_regime(rows, dev_ids, (full_selector,), config)
    full = compact(full_selector.serialize(test_ids))

    families = sorted(
        {str(rows[index].get("category")) for index in test_ids}
    )
    folds = []
    for family in families:
        fit_ids = [
            index
            for index in dev_ids
            if str(rows[index].get("category")) != family
        ]
        heldout_test_ids = [
            index
            for index in test_ids
            if str(rows[index].get("category")) == family
        ]
        fold_selector = selector(rows, config, fit_ids)
        fit_metadata = fit_regime(rows, fit_ids, (fold_selector,), config)
        fold = compact(fold_selector.serialize(heldout_test_ids))
        folds.append(
            {
                "heldout_family": family,
                "development_rows": len(fit_ids),
                "test_rows": len(heldout_test_ids),
                "fit": fit_metadata,
                **fold,
            }
        )

    return {
        "model": model,
        "tasks": ["mc"],
        "development_pairs": len({rows[index].get("pair_id") for index in dev_ids}),
        "test_pairs": len({rows[index].get("pair_id") for index in test_ids}),
        "test_refit": False,
        "alignment": alignment_audit(rows),
        "full_fit": full_fit,
        "full": full,
        "leave_one_family_out": folds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cprc_third_model_mc_v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = {
        "version": VERSION,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "experiments": {
            model: run_model(model, spec, config)
            for model, spec in config["models"].items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for model, experiment in payload["experiments"].items():
        metrics = experiment["full"]["test_metrics"]["mc"]
        print(
            model,
            f"alignment={experiment['alignment']['native_error_equals_prior_top_rate']:.3f}",
            f"MC={100 * metrics['counterfactual']['delta']:+.2f}/"
            f"{100 * metrics['commonsense']['delta']:+.2f}",
        )
        for fold in experiment["leave_one_family_out"]:
            values = fold["test_metrics"]["mc"]
            print(
                " ",
                fold["heldout_family"],
                f"{100 * values['counterfactual']['delta']:+.2f}/"
                f"{100 * values['commonsense']['delta']:+.2f}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
