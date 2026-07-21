#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


VERSION = "llava_baseline_matrix_analysis_v3"
TASKS = ("mc", "qa")
SIDES = ("counterfactual", "commonsense")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_result(path: str) -> Dict[str, Any]:
    payload = load(path)
    output = {}
    for side in SIDES:
        values = payload[side]
        output[side] = {
            "n": int(values["n"]),
            "baseline_acc": float(values["baseline_acc"]),
            "method_acc": float(values["method_acc"]),
            "delta": float(values["delta"]),
            "repairs": int(values["repairs"]),
            "harms": int(values["harms"]),
            "mcnemar_p": float(values.get("mcnemar_exact_p", 1.0)),
        }
    if isinstance(payload.get("bootstrap"), dict):
        output["counterfactual"]["delta_ci95"] = payload["bootstrap"].get(
            "delta_cf_ci95"
        )
        output["commonsense"]["delta_ci95"] = payload["bootstrap"].get(
            "delta_cs_ci95"
        )
    if isinstance(payload.get("frozen_params"), dict):
        output["frozen_params"] = payload["frozen_params"]
    return output


def revis_result(path: str, task: str) -> Dict[str, Any]:
    payload = load(path)
    output = {}
    for side in SIDES:
        values = payload["overall"][f"{task}.{side}"]
        output[side] = {
            "n": int(values["n"]),
            "baseline_acc": float(values["baseline_accuracy"]),
            "method_acc": float(values["official_revis_accuracy"]),
            "delta": float(values["delta_accuracy"]),
            "repairs": int(values["repairs"]),
            "harms": int(values["harms"]),
            "mcnemar_p": float(values["mcnemar_p"]),
            "answer_transitions": values.get("answer_transitions", {}),
        }
    return output


def cprc_result(path: str, task: str) -> Dict[str, Any]:
    payload = load(path)["result"]
    metrics = payload["evaluation_by_dataset"][f"test_{task}"]
    tests = payload["paired_tests"][f"test_{task}"]
    intervals = payload["delta_ci95"][f"test_{task}"]
    output = {}
    for side in SIDES:
        values = metrics[side]
        paired = tests[side]
        output[side] = {
            "n": int(values["n"]),
            "baseline_acc": float(values["baseline_acc"]),
            "method_acc": float(values["acc"]),
            "delta": float(values["delta"]),
            "repairs": int(paired["repairs"]),
            "harms": int(paired["harms"]),
            "mcnemar_p": float(paired["exact_mcnemar_p_two_sided"]),
            "delta_ci95": intervals[side],
        }
    return output


def validate_baselines(matrix: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    for task in TASKS:
        for side in SIDES:
            counts = [
                int(tasks[task][side]["n"])
                for tasks in matrix.values()
            ]
            if set(counts) != {230}:
                raise ValueError(
                    f"incomplete matrix for {task}/{side}: {sorted(set(counts))}"
                )
            values = [
                float(tasks[task][side]["baseline_acc"])
                for tasks in matrix.values()
            ]
            if max(values) - min(values) > 1e-12:
                raise ValueError(
                    f"baseline mismatch for {task}/{side}: {sorted(set(values))}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    for method in ("prompt", "vcd", "mfcd", "pai", "nolan"):
        for task in TASKS:
            parser.add_argument(f"--{method}-{task}", required=True)
    parser.add_argument("--revis", required=True)
    parser.add_argument("--cprc", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sources = {
        method: {task: getattr(args, f"{method}_{task}") for task in TASKS}
        for method in ("prompt", "vcd", "mfcd", "pai", "nolan")
    }
    matrix = {
        method: {task: paired_result(path) for task, path in paths.items()}
        for method, paths in sources.items()
    }
    matrix["official_revis"] = {
        task: revis_result(args.revis, task) for task in TASKS
    }
    matrix["eb_cprc"] = {task: cprc_result(args.cprc, task) for task in TASKS}
    validate_baselines(matrix)

    source_paths = [
        path for paths in sources.values() for path in paths.values()
    ] + [args.revis, args.cprc]
    payload = {
        "version": VERSION,
        "model": "LLaVA-1.6-34B-Instruct",
        "protocol": "70-pair development selection followed by frozen 230-pair CPRC-unseen evaluation",
        "sources": {
            path: sha256(Path(path)) for path in source_paths
        },
        "matrix": matrix,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("method\tMC_dCF\tMC_dCS\tQA_dCF\tQA_dCS")
    for method, tasks in matrix.items():
        print(
            f"{method}\t{tasks['mc']['counterfactual']['delta']:+.4f}"
            f"\t{tasks['mc']['commonsense']['delta']:+.4f}"
            f"\t{tasks['qa']['counterfactual']['delta']:+.4f}"
            f"\t{tasks['qa']['commonsense']['delta']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
