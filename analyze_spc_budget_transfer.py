#!/usr/bin/env python3
"""Replay one frozen SPC CS-budget operating point across paper benchmarks."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import analyze_benchmark_spectrum as spectrum
import analyze_proposal_decomposition as decomposition
from analyze_cp_vbc_bayes_path_cv import normalize_key


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen3-VL-32B-Instruct"


def paired_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "changed": metrics["overall"]["interventions"],
        "repairs": metrics["overall"]["repairs"],
        "harms": metrics["overall"]["harms"],
        "delta_cf": metrics["per_side"]["counterfactual"]["delta"],
        "delta_cs": metrics["per_side"]["commonsense"]["delta"],
        "sides": metrics["per_side"],
    }


def single_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "changed": metrics["interventions"],
        "repairs": metrics["repairs"],
        "harms": metrics["harms"],
        "delta": metrics["delta"],
        "baseline_acc": metrics["baseline_acc"],
        "acc": metrics["acc"],
        "exact_mcnemar_p_two_sided": metrics["exact_mcnemar_p_two_sided"],
    }


def load_pope(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for rel in paths:
        for source in spectrum.read_jsonl(ROOT / rel, "qa"):
            key = (str(source.get("protocol")), int(source.get("question_id")))
            if key in seen:
                raise ValueError(f"duplicate POPE row: {key}")
            seen.add(key)
            row = dict(source)
            row["_dataset"] = "ext_pope_native"
            row["_task"] = "qa"
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", default="0.10")
    parser.add_argument(
        "--output",
        default="result/paper_revision_stats/spc_budget_0.10_transfer.json",
    )
    args = parser.parse_args()

    shared = decomposition.load_shared()
    aac, _, _, backfill_config, gate_pareto, _, _ = shared
    selected = gate_pareto["experiments"][MODEL][
        "selected_by_development_cs_budget"
    ][spectrum.FAMILY][args.budget]
    frozen = selected["params"]

    rows = aac.load_model_rows(backfill_config["models"][MODEL])
    dev_rows = [row for row in rows if str(row["_dataset"]).startswith("dev_")]

    output: Dict[str, Any] = {
        "model": MODEL,
        "family": spectrum.FAMILY,
        "development_cs_budget": float(args.budget),
        "params": frozen,
        "benchmarks": {},
    }

    cdh = {}
    for task in ("mc", "qa"):
        metrics = selected["test_metrics"][f"test_{task}"]
        repair_harm = selected["test_repairs_harms"][f"test_{task}"]
        cdh[task] = {
            "changed": (
                metrics["counterfactual"]["overrides"]
                + metrics["commonsense"]["overrides"]
            ),
            "repairs": (
                repair_harm["counterfactual"]["repairs"]
                + repair_harm["commonsense"]["repairs"]
            ),
            "harms": (
                repair_harm["counterfactual"]["harms"]
                + repair_harm["commonsense"]["harms"]
            ),
            "delta_cf": metrics["counterfactual"]["delta"],
            "delta_cs": metrics["commonsense"]["delta"],
            "sides": {
                side: {
                    **metrics[side],
                    **repair_harm[side],
                    "exact_mcnemar_p_two_sided": spectrum.exact_mcnemar(
                        repair_harm[side]["repairs"],
                        repair_harm[side]["harms"],
                    ),
                }
                for side in ("counterfactual", "commonsense")
            },
        }
    output["benchmarks"]["cdh_bench"] = cdh

    vcf = spectrum.load_external(
        [spectrum.VCF_PATHS[MODEL]], "mc", "ext_vcf"
    )
    streams = spectrum.eval_external(MODEL, vcf, dev_rows, frozen)
    output["benchmarks"]["visual_counterfact"] = paired_summary(
        spectrum.benchmark_metrics(vcf, streams)["cprc"]
    )

    hallusion = spectrum.load_external(
        spectrum.HB_PATHS, "qa", "ext_hallusionbench"
    )
    streams = spectrum.eval_external(MODEL, hallusion, dev_rows, frozen)
    output["benchmarks"]["hallusionbench"] = paired_summary(
        spectrum.benchmark_metrics(hallusion, streams)["cprc"]
    )

    conflict = spectrum.load_external(
        spectrum.CONFLICTVIS_PATHS, None, "ext_conflictvis"
    )
    streams = spectrum.eval_external(MODEL, conflict, dev_rows, frozen)
    labels = [normalize_key(row.get("gt")) for row in conflict]
    conflict_tasks = {}
    for task in ("mc", "qa"):
        indices = [
            index
            for index, row in enumerate(conflict)
            if str(row.get("task")) == task
        ]
        metrics = spectrum.side_metrics(
            [labels[index] for index in indices],
            [streams["native"][index] for index in indices],
            [streams["cprc"][index] for index in indices],
        )
        conflict_tasks[task] = single_summary(metrics)
    output["benchmarks"]["conflictvis"] = conflict_tasks

    pope = load_pope(spectrum.POPE_PATHS)
    streams = spectrum.eval_external(MODEL, pope, dev_rows, frozen)
    metrics = spectrum.benchmark_metrics(pope, streams)["cprc"]["overall"]
    labels = [normalize_key(row.get("gt")) for row in pope]
    native_binary = spectrum.binary_metrics(labels, streams["native"])
    spc_binary = spectrum.binary_metrics(labels, streams["cprc"])
    output["benchmarks"]["pope"] = {
        **single_summary(metrics),
        "delta_f1": spc_binary["f1"] - native_binary["f1"],
    }

    popev2 = spectrum.load_external(
        [spectrum.POPEV2_PATH], "qa", "ext_popev2"
    )
    streams = spectrum.eval_external(MODEL, popev2, dev_rows, frozen)
    output["benchmarks"]["popev2"] = paired_summary(
        spectrum.benchmark_metrics(popev2, streams)["cprc"]
    )

    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
