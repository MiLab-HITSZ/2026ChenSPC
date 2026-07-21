#!/usr/bin/env python3
"""Benchmark-spectrum offline evaluation of the frozen CPRC learned route.

Hypothesis under test: the frozen learned MAP route (CPRC, Selective Support
Gate, map_no_laplace_or_density at development CS budget 0.04 from
result/cprc_robustness/gate_pareto_qwen_llava_v1.json; Qwen l2=0.01 cap=1.0
margin=0.3, LLaVA l2=0.01 cap=0.8 margin=0.2) transfers only to CDH-type
conflict benchmarks; on existence-hallucination benchmarks it intervenes
(approximately) never and does not damage native accuracy; the CPRC+NoLan
extension inherits NoLan's damage on existence benchmarks.

Methods replayed row by row from frozen candidate scores (pure CPU, no model
calls):

  cprc        frozen learned route: theta refit on the CDH dev70 rows with the
              frozen l2, frozen lambda cap and frozen gate (arity + conflict +
              posterior-support/margin), applied to the target rows.
  cprc_nolan  CPRC+NoLan composition: if the gated learned prediction differs
              from native keep it, otherwise backfill with the NoLan proposal
              z_A (beta=0.8) when z_A != native (same rule as
              analyze_proposal_decomposition / analyze_nolan_backfill).
  nolan       NoLan proposal z_A = argmax(logp_image - lambda_A * logp_prior)
              with lambda_A = alpha/(1+alpha), alpha = beta*(tanh(1/KL_sym)+1);
              algebraically identical to the released dynamic-score rule
              replayed by analyze_candidate_baseline_sweep._replay_nolan.

Benchmarks: CDH cpr_unseen (Qwen/LLaVA, MC+QA), VisualCounterfact (Qwen/LLaVA,
MC), HallusionBench transfer (Qwen, QA), ConflictVIS (Qwen, QA+MC, CF side
only), POPE native (Qwen, 9000 rows), POPEv2 transfer (Qwen, QA, new).

Verification: CDH replay is checked cell by cell against
result/anchored_cprc/main_test.json (learned_route / nolan / cprc_map); the
VCF learned-route replay is checked against the frozen summary of
result/cprc_external/visual_counterfact_qwen_llava_v1.json; NoLan replays are
checked against result/nolan_external_transfer/*.json; the POPE-native learned
replay is checked against result/cprc_external/pope_native_full_cdh_frozen_bf16_v1.json.

Re-run:  cdh-bench-env/bin/python analyze_benchmark_spectrum.py

Outputs:
  result/paper_revision_stats/benchmark_spectrum.json
  result/paper_revision_stats/benchmark_spectrum_table.tex
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import analyze_cprc_gate_pareto as gp
import analyze_hierarchical_eb_lambda_cv as eb
import analyze_proposal_decomposition as apd
from analyze_cp_vbc_bayes_path_cv import normalize_key, read_jsonl
from analyze_nolan_anchored_cprc import nolan_lambda
from analyze_pope_native_cprc import binary_metrics, exact_mcnemar

ROOT = Path(__file__).resolve().parent
OUTPUT_JSON = ROOT / "result" / "paper_revision_stats" / "benchmark_spectrum.json"
OUTPUT_TEX = ROOT / "result" / "paper_revision_stats" / "benchmark_spectrum_table.tex"

FAMILY = "map_no_laplace_or_density"
QUADRATURE_POINTS = 64
NOLAN_BETA = 0.8
SIDES = ("counterfactual", "commonsense")

VCF_PATHS = {
    "Qwen3-VL-32B-Instruct": "result/visual_counterfact_frozen_v1/qwen/results.jsonl",
    "LLaVA-1.6-34B": "result/visual_counterfact_frozen_v1/llava/results.jsonl",
}
HB_PATHS = [
    "result/hallusionbench_qa_transfer_full_32b_bf16_native_candidate_scores/shard0.jsonl",
    "result/hallusionbench_qa_transfer_full_32b_bf16_native_candidate_scores/shard1.jsonl",
]
CONFLICTVIS_PATHS = [
    "result/cprc_external/conflictvis_full_bf16_shard0.jsonl",
    "result/cprc_external/conflictvis_full_bf16_shard1.jsonl",
]
POPE_PATHS = [
    "result/pope_native_qwen32b_bf16/shard0.jsonl",
    "result/pope_native_qwen32b_bf16/shard1.jsonl",
]
POPEV2_PATH = (
    "result/popev2_qa_transfer_32b_instruct/"
    "Qwen3-VL-32B-Instruct-transformers-maxtok48/results.jsonl"
)

VCF_FROZEN_JSON = "result/cprc_external/visual_counterfact_qwen_llava_v1.json"
POPE_CPRC_JSON = "result/cprc_external/pope_native_full_cdh_frozen_bf16_v1.json"
HB_CPRC_JSON = "result/cprc_external/hallusionbench_full_cdh_frozen_bf16_native_v1.json"
CONFLICTVIS_CPRC_JSON = "result/cprc_external/conflictvis_full_frozen_bf16_v1.json"
NOLAN_JSONS = {
    "vcf_qwen": "result/nolan_external_transfer/visual_counterfact_qwen_nolan_v1.json",
    "vcf_llava": "result/nolan_external_transfer/visual_counterfact_llava_nolan_v1.json",
    "hallusionbench": "result/nolan_external_transfer/hallusionbench_qwen_nolan_v1.json",
    "conflictvis": "result/nolan_external_transfer/conflictvis_nolan_v1.json",
    "pope_native": "result/nolan_external_transfer/pope_nolan_v1.json",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external(paths: Sequence[str], task: Any, dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for rel in paths:
        for source in read_jsonl(ROOT / rel, task):
            row = dict(source)
            key = (str(row.get("pair_id")), str(row.get("side")), str(row.get("_task") or row.get("task")))
            if key in seen:
                raise ValueError(f"duplicate external row: {key}")
            seen.add(key)
            row["_dataset"] = dataset
            row["_task"] = str(row.get("task") or task)
            rows.append(row)
    return rows


def nolan_predictions(rows: Sequence[Dict[str, Any]], beta: float = NOLAN_BETA) -> List[str]:
    """NoLan proposal stream z_A (identical to _replay_nolan with beta=0.8)."""
    output = []
    for row in rows:
        keys, image, prior = eb.candidate_arrays(dict(row))
        lam = nolan_lambda(row, beta)
        output.append(str(keys[int(np.argmax(image - float(lam) * prior))]))
    return output


def frozen_route(
    rows: Sequence[Dict[str, Any]],
    geometry: Any,
    frozen: Mapping[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Replay the frozen learned-route operating point on every row."""
    l2 = float(frozen["l2"])
    cap = float(frozen["lambda_cap"])
    theta, covariance, train_loss = eb.fit_lambda_posterior(
        rows, geometry.contexts, geometry.train_ids, l2
    )
    details = eb.policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        theta,
        np.zeros_like(covariance),
        lambda_cap=cap,
        quadrature_points=QUADRATURE_POINTS,
        support_distances=np.zeros(len(rows)),
        arity_supported=geometry.arity_supported,
    )
    learned = gp.apply_gate_family(rows, details, frozen["gate"], FAMILY)
    return learned, details


def legacy_density_gated_interventions(
    dev_rows: Sequence[Dict[str, Any]],
    ext_rows: Sequence[Dict[str, Any]],
    params: Mapping[str, Any],
) -> int:
    """Reproduce the legacy run_cv frozen_transfer selection on external rows.

    The earlier external artifacts (POPE native, ConflictVIS) were produced by
    analyze_hierarchical_eb_lambda_cv.run_cv, which re-selects on the dev rows
    and keeps the Mahalanobis density gate (Laplace covariance, real support
    distances, bprc_full-style gate). The frozen map_no_laplace_or_density
    operating point removes that density gate; this replay reproduces the
    legacy intervention counts exactly and thereby explains any difference.
    """
    rows = [dict(row) for row in dev_rows] + [dict(row) for row in ext_rows]
    geometry = gp.prepare_geometry(rows)
    theta, covariance, _ = eb.fit_lambda_posterior(
        rows, geometry.contexts, geometry.train_ids, float(params["l2"])
    )
    details = eb.policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        theta,
        covariance,
        lambda_cap=float(params["lambda_cap"]),
        quadrature_points=QUADRATURE_POINTS,
        support_distances=geometry.support_distances,
        arity_supported=geometry.arity_supported,
    )
    predictions = gp.apply_gate_family(rows, details, params["gate"], "bprc_full")
    n_dev = len(dev_rows)
    native = [eb.resolved_baseline_key(dict(row)) for row in rows]
    return sum(a != b for a, b in zip(native[n_dev:], predictions[n_dev:]))


def eval_external(
    model: str,
    ext_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    frozen: Mapping[str, Any],
) -> Dict[str, List[str]]:
    """Fit on CDH dev rows only; replay all three method streams on ext rows."""
    rows = [dict(row) for row in dev_rows] + ext_rows
    geometry = gp.prepare_geometry(rows)
    learned, _ = frozen_route(rows, geometry, frozen)
    native = [eb.resolved_baseline_key(dict(row)) for row in rows]
    nolan = nolan_predictions(rows)
    cprc_nolan = [
        learned[i] if learned[i] != native[i] else nolan[i] for i in range(len(rows))
    ]
    n_dev = len(dev_rows)
    return {
        "native": native[n_dev:],
        "cprc": learned[n_dev:],
        "nolan": nolan[n_dev:],
        "cprc_nolan": cprc_nolan[n_dev:],
    }


def side_metrics(
    labels: Sequence[str], baseline: Sequence[str], method: Sequence[str]
) -> Dict[str, Any]:
    n = len(labels)
    baseline_correct = sum(b == g for g, b in zip(labels, baseline))
    correct = sum(m == g for g, m in zip(labels, method))
    interventions = sum(b != m for b, m in zip(baseline, method))
    repairs = sum(b != g and m == g for g, b, m in zip(labels, baseline, method))
    harms = sum(b == g and m != g for g, b, m in zip(labels, baseline, method))
    return {
        "n": n,
        "baseline_correct": baseline_correct,
        "correct": correct,
        "baseline_acc": baseline_correct / n if n else 0.0,
        "acc": correct / n if n else 0.0,
        "delta": (correct - baseline_correct) / n if n else 0.0,
        "interventions": interventions,
        "repairs": repairs,
        "harms": harms,
        "exact_mcnemar_p_two_sided": exact_mcnemar(repairs, harms),
    }


def benchmark_metrics(
    rows: Sequence[Dict[str, Any]], streams: Mapping[str, List[str]]
) -> Dict[str, Any]:
    """Per-side and overall metrics for every method stream."""
    labels = [normalize_key(row.get("gt")) for row in rows]
    baseline = [str(v) for v in streams["native"]]
    out: Dict[str, Any] = {}
    for method in ("cprc", "cprc_nolan", "nolan"):
        predictions = [str(v) for v in streams[method]]
        per_side = {}
        for side in SIDES:
            idx = [i for i, row in enumerate(rows) if str(row.get("side")) == side]
            if not idx:
                continue
            per_side[side] = side_metrics(
                [labels[i] for i in idx],
                [baseline[i] for i in idx],
                [predictions[i] for i in idx],
            )
        out[method] = {
            "overall": side_metrics(labels, baseline, predictions),
            "per_side": per_side,
        }
    return out


def joint_deltas(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    per_side = metrics["per_side"]
    return {
        "delta_cf": per_side.get("counterfactual", {}).get("delta"),
        "delta_cs": per_side.get("commonsense", {}).get("delta"),
        "interventions": metrics["overall"]["interventions"],
        "delta_overall": metrics["overall"]["delta"],
    }


# --------------------------------------------------------------------------- #
# CDH cpr_unseen (anchored pipeline, verified against main_test.json)
# --------------------------------------------------------------------------- #

def run_cdh(
    model: str,
    spec: Mapping[str, Any],
    shared: Tuple[Any, Any, Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str],
) -> Dict[str, Any]:
    aac, eb_mod, gp_mod, backfill_config, gate_pareto, main_test, budget = shared
    replay = apd.frozen_replay(model, spec, aac, eb_mod, gp_mod, gate_pareto, budget)
    rows = replay["rows"]
    geometry = replay["geometry"]
    reference = replay["reference"]
    streams = {
        "native": [str(v) for v in reference["native"]],
        "cprc": [str(v) for v in reference["learned_route"]],
        "nolan": [str(v) for v in reference["nolan"]],
        "cprc_nolan": [str(v) for v in reference["cprc_map"]],
    }
    labels = [normalize_key(row.get("gt")) for row in rows]
    test_ids = list(geometry.test_ids)

    methods: Dict[str, Any] = {}
    for method in ("cprc", "cprc_nolan", "nolan"):
        per_side: Dict[str, Any] = {}
        for side in SIDES:
            idx = [i for i in test_ids if str(rows[i].get("side")) == side]
            per_side[side] = side_metrics(
                [labels[i] for i in idx],
                [streams["native"][i] for i in idx],
                [streams[method][i] for i in idx],
            )
        methods[method] = {
            "overall": side_metrics(
                [labels[i] for i in test_ids],
                [streams["native"][i] for i in test_ids],
                [streams[method][i] for i in test_ids],
            ),
            "per_side": per_side,
        }

    # Cell-level verification against the frozen main_test.json summaries.
    expected = main_test["experiments"][model]["methods"]
    report_names = {"cprc": "learned_route", "cprc_nolan": "cprc_map", "nolan": "nolan"}
    checks: Dict[str, Any] = {}
    all_match = bool(replay["verification"]["matches"])
    for method, report_name in report_names.items():
        report = aac.method_report(rows, streams[method], test_ids, bootstrap=0)
        for cell in ("test_mc", "test_qa"):
            for side in SIDES:
                got = report[cell][side]
                want = expected[report_name][cell][side]
                match = all(
                    got[key] == want[key]
                    for key in ("correct", "overrides", "baseline_correct", "repairs", "harms")
                )
                all_match &= match
                checks[f"{report_name}/{cell}/{side}"] = {
                    "expected": {key: want[key] for key in ("correct", "overrides", "baseline_correct", "repairs", "harms")},
                    "recomputed": {key: got[key] for key in ("correct", "overrides", "baseline_correct", "repairs", "harms")},
                    "matches": match,
                }
    return {
        "n_rows": len(test_ids),
        "methods": methods,
        "verification": {
            "learned_route_replay_identical_to_anchored_reference": replay["verification"],
            "main_test_cells": checks,
            "matches_frozen_main_test": bool(all_match),
        },
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def fmt_delta(value: Any) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):+.1f}"


def main() -> int:
    shared = apd.load_shared()
    aac, eb_mod, gp_mod, backfill_config, gate_pareto, main_test, budget = shared
    assert budget == "0.04", budget

    frozen_params = {
        model: gate_pareto["experiments"][model]["selected_by_development_cs_budget"][
            FAMILY
        ][budget]["params"]
        for model in backfill_config["models"]
    }
    cdh_rows = {
        model: aac.load_model_rows(spec)
        for model, spec in backfill_config["models"].items()
    }
    dev_rows = {
        model: [row for row in rows if str(row["_dataset"]).startswith("dev_")]
        for model, rows in cdh_rows.items()
    }

    input_files: Dict[str, Any] = {}

    def register(paths: Sequence[str]) -> List[Dict[str, Any]]:
        entries = []
        for rel in paths:
            path = ROOT / rel
            entries.append(
                {
                    "path": rel,
                    "rows": sum(1 for _ in open(path, "rb")),
                    "sha256": sha256(path),
                }
            )
        return entries

    benchmarks: Dict[str, Any] = {}

    # ---------------- CDH cpr_unseen (in-distribution blind test) ----------
    for model, spec in backfill_config["models"].items():
        key = f"cdh_cpr_unseen/{model}"
        result = run_cdh(model, spec, shared)
        benchmarks[key] = {
            "benchmark": "CDH cpr_unseen",
            "type": "CDH-type conflict (in-distribution blind test)",
            "model": model,
            "task": "mc+qa",
            "n_rows": result["n_rows"],
            "n_pairs": result["n_rows"] // 2,
            "methods": result["methods"],
            "verification": result["verification"],
        }
    input_files["cdh"] = {
        "dev_mc": backfill_config["models"]["Qwen3-VL-32B-Instruct"]["dev_mc"],
        "dev_qa": backfill_config["models"]["Qwen3-VL-32B-Instruct"]["dev_qa"],
        "test": backfill_config["models"]["Qwen3-VL-32B-Instruct"]["test"],
        "llava_dev_mc": backfill_config["models"]["LLaVA-1.6-34B"]["dev_mc"],
        "llava_dev_qa": backfill_config["models"]["LLaVA-1.6-34B"]["dev_qa"],
        "llava_test": backfill_config["models"]["LLaVA-1.6-34B"]["test"],
        "reference_summary": "result/anchored_cprc/main_test.json",
    }

    # ---------------- VisualCounterfact (Qwen + LLaVA, MC) -----------------
    vcf_frozen = json.loads((ROOT / VCF_FROZEN_JSON).read_text(encoding="utf-8"))
    vcf_nolan_refs = {
        "Qwen3-VL-32B-Instruct": json.loads((ROOT / NOLAN_JSONS["vcf_qwen"]).read_text(encoding="utf-8")),
        "LLaVA-1.6-34B": json.loads((ROOT / NOLAN_JSONS["vcf_llava"]).read_text(encoding="utf-8")),
    }
    for model, rel in VCF_PATHS.items():
        ext = load_external([rel], "mc", "ext_vcf")
        streams = eval_external(model, ext, dev_rows[model], frozen_params[model])
        metrics = benchmark_metrics(ext, streams)
        ref = vcf_frozen["experiments"][model]["selected"][FAMILY][budget]["by_stratum"]["overall"]
        nref = vcf_nolan_refs[model]
        checks: Dict[str, Any] = {"cprc_vs_frozen_summary": {}, "nolan_vs_frozen_summary": {}}
        cprc_ok = True
        for side in SIDES:
            got = metrics["cprc"]["per_side"][side]
            want_m = ref["metrics"][side]
            want_p = ref["paired"][side]
            match = (
                got["correct"] == want_m["correct"]
                and got["baseline_correct"] == want_m["baseline_correct"]
                and got["interventions"] == want_m["overrides"]
                and got["repairs"] == want_p["repairs"]
                and got["harms"] == want_p["harms"]
            )
            cprc_ok &= match
            checks["cprc_vs_frozen_summary"][side] = {
                "expected": {
                    "correct": want_m["correct"],
                    "overrides": want_m["overrides"],
                    "repairs": want_p["repairs"],
                    "harms": want_p["harms"],
                },
                "recomputed": {
                    "correct": got["correct"],
                    "overrides": got["interventions"],
                    "repairs": got["repairs"],
                    "harms": got["harms"],
                },
                "matches": match,
            }
        nolan_ok = True
        for side in SIDES:
            got = metrics["nolan"]["per_side"][side]
            want = nref[side]
            match = got["repairs"] == want["repairs"] and got["harms"] == want["harms"] and abs(
                got["delta"] - want["delta"]
            ) < 1e-12
            nolan_ok &= match
            checks["nolan_vs_frozen_summary"][side] = {
                "expected": {"repairs": want["repairs"], "harms": want["harms"], "delta": want["delta"]},
                "recomputed": {"repairs": got["repairs"], "harms": got["harms"], "delta": got["delta"]},
                "matches": match,
            }
        benchmarks[f"visual_counterfact/{model}"] = {
            "benchmark": "VisualCounterfact",
            "type": "CDH-type conflict (transfer)",
            "model": model,
            "task": "mc",
            "n_rows": len(ext),
            "n_pairs": len(ext) // 2,
            "methods": metrics,
            "verification": {
                "reference_cprc": VCF_FROZEN_JSON,
                "reference_nolan": NOLAN_JSONS["vcf_qwen" if model.startswith("Qwen") else "vcf_llava"],
                **checks,
                "matches_frozen_summary": bool(cprc_ok and nolan_ok),
            },
        }
    input_files["visual_counterfact"] = register(list(VCF_PATHS.values()))

    # ---------------- HallusionBench transfer (Qwen, QA) -------------------
    hb_ext = load_external(HB_PATHS, "qa", "ext_hallusionbench")
    hb_streams = eval_external("Qwen3-VL-32B-Instruct", hb_ext, dev_rows["Qwen3-VL-32B-Instruct"], frozen_params["Qwen3-VL-32B-Instruct"])
    hb_metrics = benchmark_metrics(hb_ext, hb_streams)
    hb_nolan_ref = json.loads((ROOT / NOLAN_JSONS["hallusionbench"]).read_text(encoding="utf-8"))
    hb_high_repair = json.loads((ROOT / HB_CPRC_JSON).read_text(encoding="utf-8"))
    hb_nolan_ok = True
    hb_checks: Dict[str, Any] = {}
    for side in SIDES:
        got = hb_metrics["nolan"]["per_side"][side]
        want = hb_nolan_ref[side]
        match = got["repairs"] == want["repairs"] and got["harms"] == want["harms"] and abs(
            got["delta"] - want["delta"]
        ) < 1e-12
        hb_nolan_ok &= match
        hb_checks[side] = {
            "expected": {"repairs": want["repairs"], "harms": want["harms"], "delta": want["delta"]},
            "recomputed": {"repairs": got["repairs"], "harms": got["harms"], "delta": got["delta"]},
            "matches": match,
        }
    benchmarks["hallusionbench/Qwen3-VL-32B-Instruct"] = {
        "benchmark": "HallusionBench (paired transfer)",
        "type": "CDH-type conflict (transfer)",
        "model": "Qwen3-VL-32B-Instruct",
        "task": "qa",
        "n_rows": len(hb_ext),
        "n_pairs": len(hb_ext) // 2,
        "methods": hb_metrics,
        "existing_high_repair_reference": {
            "path": HB_CPRC_JSON,
            "note": (
                "Previous paper number: run_cv frozen_transfer re-selection "
                "(l2=0.01, cap=1.0, margin=0.0 + density gate), not the frozen "
                "map_no_laplace_or_density@0.04 operating point."
            ),
            "by_dataset": hb_high_repair["result"]["by_dataset"]["hallusion_full"],
        },
        "verification": {
            "reference_nolan": NOLAN_JSONS["hallusionbench"],
            "nolan_vs_frozen_summary": hb_checks,
            "matches_frozen_summary": bool(hb_nolan_ok),
        },
    }
    input_files["hallusionbench"] = register(HB_PATHS)

    # ---------------- ConflictVIS (Qwen, QA + MC, CF side only) -------------
    cv_ext = load_external(CONFLICTVIS_PATHS, None, "ext_conflictvis")
    cv_streams = eval_external("Qwen3-VL-32B-Instruct", cv_ext, dev_rows["Qwen3-VL-32B-Instruct"], frozen_params["Qwen3-VL-32B-Instruct"])
    cv_metrics = benchmark_metrics(cv_ext, cv_streams)
    # per-task slices
    cv_tasks: Dict[str, Any] = {}
    labels_cv = [normalize_key(row.get("gt")) for row in cv_ext]
    for task in ("qa", "mc"):
        idx = [i for i, row in enumerate(cv_ext) if str(row.get("task")) == task]
        cv_tasks[task] = {
            method: side_metrics(
                [labels_cv[i] for i in idx],
                [cv_streams["native"][i] for i in idx],
                [cv_streams[method][i] for i in idx],
            )
            for method in ("cprc", "cprc_nolan", "nolan")
        }
    cv_nolan_ref = json.loads((ROOT / NOLAN_JSONS["conflictvis"]).read_text(encoding="utf-8"))
    cv_cprc_ref = json.loads((ROOT / CONFLICTVIS_CPRC_JSON).read_text(encoding="utf-8"))
    cv_nolan_ok = (
        cv_metrics["nolan"]["overall"]["interventions"]
        == cv_nolan_ref["metrics"]["overall"]["interventions"]
        and cv_metrics["nolan"]["overall"]["repairs"] == cv_nolan_ref["metrics"]["overall"]["repairs"]
        and cv_metrics["nolan"]["overall"]["harms"] == cv_nolan_ref["metrics"]["overall"]["harms"]
    )
    cv_cprc_ref_overall = cv_cprc_ref["metrics"]["overall"]["eb_cprc"]
    cv_legacy_params = cv_cprc_ref["eb_fit_selection"][0]["params"]
    cv_legacy_interventions = legacy_density_gated_interventions(
        dev_rows["Qwen3-VL-32B-Instruct"], cv_ext, cv_legacy_params
    )
    cv_legacy_ok = cv_legacy_interventions == cv_cprc_ref_overall["interventions"]
    benchmarks["conflictvis/Qwen3-VL-32B-Instruct"] = {
        "benchmark": "ConflictVIS",
        "type": "CDH-type conflict (transfer, counterfactual side only)",
        "model": "Qwen3-VL-32B-Instruct",
        "task": "qa+mc",
        "n_rows": len(cv_ext),
        "n_pairs": len(cv_ext),
        "methods": cv_metrics,
        "per_task": cv_tasks,
        "existing_cprc_reference": {
            "path": CONFLICTVIS_CPRC_JSON,
            "note": (
                "Previous number used run_cv frozen_transfer re-selection "
                "(margin=0.0 + Mahalanobis density gate); the frozen "
                "map_no_laplace_or_density@0.04 operating point removes the "
                "density gate, which is why the intervention counts differ."
            ),
            "overall": cv_cprc_ref_overall,
            "frozen_operating_point_interventions": cv_metrics["cprc"]["overall"]["interventions"],
        },
        "verification": {
            "reference_nolan": NOLAN_JSONS["conflictvis"],
            "nolan_overall_matches_frozen_summary": bool(cv_nolan_ok),
            "legacy_density_gated_replay": {
                "params": {
                    "l2": cv_legacy_params["l2"],
                    "lambda_cap": cv_legacy_params["lambda_cap"],
                    "gate": cv_legacy_params["gate"],
                },
                "recomputed_interventions": cv_legacy_interventions,
                "expected_interventions": cv_cprc_ref_overall["interventions"],
                "matches_existing_artifact": bool(cv_legacy_ok),
            },
        },
    }
    input_files["conflictvis"] = register(CONFLICTVIS_PATHS)

    # ---------------- POPE native (Qwen, 9000 rows, existence) --------------
    pope_rows: List[Dict[str, Any]] = []
    seen_pope = set()
    for rel in POPE_PATHS:
        for source in read_jsonl(ROOT / rel, "qa"):
            key = (str(source.get("protocol")), int(source.get("question_id")))
            if key in seen_pope:
                raise ValueError(f"duplicate POPE row: {key}")
            seen_pope.add(key)
            row = dict(source)
            row["_dataset"] = "ext_pope_native"
            row["_task"] = "qa"
            pope_rows.append(row)
    pope_streams = eval_external("Qwen3-VL-32B-Instruct", pope_rows, dev_rows["Qwen3-VL-32B-Instruct"], frozen_params["Qwen3-VL-32B-Instruct"])
    pope_metrics = benchmark_metrics(pope_rows, pope_streams)
    pope_labels = [normalize_key(row.get("gt")) for row in pope_rows]
    pope_binary = {
        method: binary_metrics(pope_labels, [str(v) for v in pope_streams[method]])
        for method in ("native", "cprc", "cprc_nolan", "nolan")
    }
    pope_cprc_artifact = json.loads((ROOT / POPE_CPRC_JSON).read_text(encoding="utf-8"))
    pope_cprc_ref = pope_cprc_artifact["results"]["overall"]
    pope_nolan_ref = json.loads((ROOT / NOLAN_JSONS["pope_native"]).read_text(encoding="utf-8"))["results"]["overall"]
    pope_legacy_params = pope_cprc_artifact["fit_selection"][0]["params"]
    pope_legacy_interventions = legacy_density_gated_interventions(
        dev_rows["Qwen3-VL-32B-Instruct"], pope_rows, pope_legacy_params
    )
    pope_legacy_ok = pope_legacy_interventions == pope_cprc_ref["interventions"]
    pope_nolan_ok = (
        pope_metrics["nolan"]["overall"]["interventions"] == pope_nolan_ref["interventions"]
        and pope_metrics["nolan"]["overall"]["repairs"] == pope_nolan_ref["paired_test"]["repairs"]
        and pope_metrics["nolan"]["overall"]["harms"] == pope_nolan_ref["paired_test"]["harms"]
    )
    benchmarks["pope_native/Qwen3-VL-32B-Instruct"] = {
        "benchmark": "POPE native (COCO random/popular/adversarial)",
        "type": "existence hallucination",
        "model": "Qwen3-VL-32B-Instruct",
        "task": "qa",
        "n_rows": len(pope_rows),
        "n_pairs": len(pope_rows),
        "methods": pope_metrics,
        "binary_metrics": pope_binary,
        "existing_cprc_reference": {
            "path": POPE_CPRC_JSON,
            "note": (
                "Existing artifact reports 0 interventions under the run_cv "
                "frozen_transfer selection (margin=0.0 + Mahalanobis density "
                "gate); the frozen map_no_laplace_or_density@0.04 operating "
                "point removes the density gate and intervenes on "
                f"{pope_metrics['cprc']['overall']['interventions']} of 9000 rows "
                "(near-zero, accuracy-neutral)."
            ),
            "overall": {
                "interventions": pope_cprc_ref["interventions"],
                "delta_accuracy": pope_cprc_ref["delta_accuracy"],
            },
        },
        "verification": {
            "reference_cprc": POPE_CPRC_JSON,
            "legacy_density_gated_replay": {
                "params": {
                    "l2": pope_legacy_params["l2"],
                    "lambda_cap": pope_legacy_params["lambda_cap"],
                    "gate": pope_legacy_params["gate"],
                },
                "recomputed_interventions": pope_legacy_interventions,
                "expected_interventions": pope_cprc_ref["interventions"],
                "matches_existing_artifact": bool(pope_legacy_ok),
            },
            "reference_nolan": NOLAN_JSONS["pope_native"],
            "nolan_overall_matches_frozen_summary": bool(pope_nolan_ok),
        },
    }
    input_files["pope_native"] = register(POPE_PATHS)

    # ---------------- POPEv2 transfer (Qwen, QA, new) -----------------------
    popev2_ext = load_external([POPEV2_PATH], "qa", "ext_popev2")
    rows_missing_logp_image = 0
    rows_missing_logp_prior = 0
    rows_missing_candidates = 0
    for row in popev2_ext:
        candidates = (row.get("cp_vbc") or {}).get("candidates") or []
        if not candidates:
            rows_missing_candidates += 1
            continue
        if any(c.get("logp_image") is None for c in candidates):
            rows_missing_logp_image += 1
        if any(c.get("logp_prior") is None for c in candidates):
            rows_missing_logp_prior += 1
    popev2_streams = eval_external("Qwen3-VL-32B-Instruct", popev2_ext, dev_rows["Qwen3-VL-32B-Instruct"], frozen_params["Qwen3-VL-32B-Instruct"])
    popev2_metrics = benchmark_metrics(popev2_ext, popev2_streams)
    popev2_labels = [normalize_key(row.get("gt")) for row in popev2_ext]
    popev2_binary = {
        method: binary_metrics(popev2_labels, [str(v) for v in popev2_streams[method]])
        for method in ("native", "cprc", "cprc_nolan", "nolan")
    }
    benchmarks["popev2/Qwen3-VL-32B-Instruct"] = {
        "benchmark": "POPEv2 (paired transfer)",
        "type": "existence hallucination (transfer)",
        "model": "Qwen3-VL-32B-Instruct",
        "task": "qa",
        "n_rows": len(popev2_ext),
        "n_pairs": len(popev2_ext) // 2,
        "methods": popev2_metrics,
        "binary_metrics": popev2_binary,
        "row_validation": {
            "rows": len(popev2_ext),
            "rows_missing_candidates": rows_missing_candidates,
            "rows_missing_logp_image": rows_missing_logp_image,
            "rows_missing_logp_prior": rows_missing_logp_prior,
        },
    }
    input_files["popev2"] = register([POPEV2_PATH])

    # ---------------- spectrum summary table -------------------------------
    spectrum: List[Dict[str, Any]] = []
    for key, bench in benchmarks.items():
        row: Dict[str, Any] = {
            "benchmark": bench["benchmark"],
            "type": bench["type"],
            "model": bench["model"],
            "n_rows": bench["n_rows"],
        }
        for method in ("cprc", "cprc_nolan", "nolan"):
            row[method] = joint_deltas(bench["methods"][method])
        spectrum.append(row)

    verification_all = {
        key: bench.get("verification", {})
        for key, bench in benchmarks.items()
    }

    def _check_passed(ver: Mapping[str, Any]) -> bool:
        flags = []
        for name in (
            "matches_frozen_main_test",
            "matches_frozen_summary",
            "nolan_overall_matches_frozen_summary",
        ):
            if name in ver:
                flags.append(bool(ver[name]))
        legacy = ver.get("legacy_density_gated_replay")
        if legacy is not None:
            flags.append(bool(legacy["matches_existing_artifact"]))
        return all(flags) if flags else True

    all_ok = all(_check_passed(ver) for ver in verification_all.values())

    payload = {
        "version": "benchmark_spectrum_v1",
        "generated_by": "analyze_benchmark_spectrum.py",
        "operating_point": {
            "family": FAMILY,
            "development_cs_budget": float(budget),
            "source": "result/cprc_robustness/gate_pareto_qwen_llava_v1.json",
            "params": {
                model: {
                    "l2": float(params["l2"]),
                    "lambda_cap": float(params["lambda_cap"]),
                    "gate": {
                        k: (v if math.isfinite(float(v)) else "inf")
                        for k, v in params["gate"].items()
                    },
                }
                for model, params in frozen_params.items()
            },
            "quadrature_points": QUADRATURE_POINTS,
            "nolan_beta": NOLAN_BETA,
            "cprc_nolan_rule": "g_M=1 -> learned proposal; else NoLan proposal z_A when z_A != native",
        },
        "inputs": input_files,
        "reference_artifacts": {
            "cdh_main_test": "result/anchored_cprc/main_test.json",
            "vcf_cprc": VCF_FROZEN_JSON,
            "pope_native_cprc": POPE_CPRC_JSON,
            "hallusionbench_cprc_high_repair": HB_CPRC_JSON,
            "conflictvis_cprc": CONFLICTVIS_CPRC_JSON,
            "nolan": NOLAN_JSONS,
        },
        "spectrum": spectrum,
        "benchmarks": benchmarks,
        "verification_passed": bool(all_ok),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # ---------------- LaTeX fragment ----------------------------------------
    def delta_cell(deltas: Mapping[str, Any]) -> str:
        cf, cs = deltas["delta_cf"], deltas["delta_cs"]
        if cf is not None and cs is not None:
            return f"{fmt_delta(cf)} / {fmt_delta(cs)}"
        return fmt_delta(deltas["delta_overall"])

    lines = [
        "% Auto-generated by analyze_benchmark_spectrum.py; booktabs style.",
        "% Deltas are accuracy changes vs the native baseline in percentage points (CF / CS).",
        "\\begin{tabular}{lllr rr c rr c rr c}",
        "\\toprule",
        " & & & & \\multicolumn{3}{c}{CPRC (learned route)} & \\multicolumn{3}{c}{CPRC+NoLan} & \\multicolumn{3}{c}{NoLan} \\\\",
        "\\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}",
        "Benchmark & Model & Type & $N$ & int. & rep./harm & $\\Delta$CF / $\\Delta$CS & int. & rep./harm & $\\Delta$CF / $\\Delta$CS & int. & rep./harm & $\\Delta$CF / $\\Delta$CS \\\\",
        "\\midrule",
    ]
    model_short = {
        "Qwen3-VL-32B-Instruct": "Qwen3-VL-32B",
        "LLaVA-1.6-34B": "LLaVA-1.6-34B",
    }
    type_short = {
        "CDH-type conflict (in-distribution blind test)": "conflict (in-dist.)",
        "CDH-type conflict (transfer)": "conflict (transfer)",
        "CDH-type conflict (transfer, counterfactual side only)": "conflict (transfer)",
        "existence hallucination": "existence",
        "existence hallucination (transfer)": "existence (transfer)",
    }
    benchmark_short = {
        "CDH cpr_unseen": "CDH cpr\\_unseen",
        "POPE native (COCO random/popular/adversarial)": "POPE native",
        "HallusionBench (paired transfer)": "HallusionBench",
        "POPEv2 (paired transfer)": "POPEv2",
    }
    for row in spectrum:
        cells = [
            benchmark_short.get(row["benchmark"], row["benchmark"]),
            model_short.get(row["model"], row["model"]),
            type_short.get(row["type"], row["type"]),
            str(row["n_rows"]),
        ]
        for method in ("cprc", "cprc_nolan", "nolan"):
            m = row[method]
            overall = benchmarks[
                [k for k in benchmarks if benchmarks[k]["benchmark"] == row["benchmark"] and benchmarks[k]["model"] == row["model"]][0]
            ]["methods"][method]["overall"]
            cells.append(str(m["interventions"]))
            cells.append(f"{overall['repairs']}/{overall['harms']}")
            cells.append(delta_cell(m))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUTPUT_TEX.relative_to(ROOT)}")
    print(f"verification_passed: {all_ok}")
    for row in spectrum:
        print(
            f"{row['benchmark']:45s} {row['model']:22s} N={row['n_rows']:5d} "
            f"CPRC int={row['cprc']['interventions']:4d} dCF={fmt_delta(row['cprc']['delta_cf'])} dCS={fmt_delta(row['cprc']['delta_cs'])} | "
            f"CPRC+NoLan int={row['cprc_nolan']['interventions']:4d} dCF={fmt_delta(row['cprc_nolan']['delta_cf'])} dCS={fmt_delta(row['cprc_nolan']['delta_cs'])} | "
            f"NoLan int={row['nolan']['interventions']:4d} dCF={fmt_delta(row['nolan']['delta_cf'])} dCS={fmt_delta(row['nolan']['delta_cs'])}"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
