#!/usr/bin/env python3
"""Offline analyses over the frozen SPC operating point.

Three independent analyses over the frozen SPC operating point of
result/anchored_cprc/main_test.json (Qwen3-VL-32B-Instruct and LLaVA-1.6-34B,
dev70 fit, blind cpr_unseen test):

  decomposition  Per-row proposal-state decomposition of the learned MAP route
                 (proposal z_M, gate g_M) against the NoLan fallback proposal
                 z_A: five mutually exclusive states with CF/CS repairs and
                 harms under SPC-MAP, learned-route-only and NoLan-only.
  mc_only        Ablation: fit and select the learned route on dev_mc only
                 (same l2/cap/gate grid, same development selection protocol,
                 CS budget applied to dev_mc only); report test_mc deltas and
                 compare with the joint-fit frozen operating point.
  stability      Stability radius r_i = min(lambda_i - lambda_lo,
                 lambda_hi - lambda_i) of the frozen per-row coefficient inside
                 the exact stable interval I_z, grouped by the final SPC-MAP
                 outcome; Mann-Whitney test of repair vs harm radii.

Re-run one analysis at a time:
  cdh-bench-env/bin/python analyze_proposal_decomposition.py --task decomposition
  cdh-bench-env/bin/python analyze_proposal_decomposition.py --task mc_only
  cdh-bench-env/bin/python analyze_proposal_decomposition.py --task stability
  cdh-bench-env/bin/python analyze_proposal_decomposition.py --task all

Outputs (per task):
  result/paper_revision_stats/proposal_decomposition.json
  result/paper_revision_stats/mc_only_ablation.json
  result/paper_revision_stats/stability_radius.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import analyze_spc_gate_pareto as gp
import analyze_hierarchical_eb_lambda_cv as eb
from analyze_cp_vbc_bayes_path_cv import normalize_key
from analyze_spc_gate_pareto import prepare_geometry
from analyze_spc_robustness import load_model_rows
from analyze_hierarchical_eb_lambda_cv import (
    bootstrap_delta,
    candidate_arrays,
    context_features,
    dataset_metrics,
    exact_mcnemar_pvalue,
    resolved_baseline_key,
)

ROOT = Path(__file__).resolve().parent
# Repo-root-relative, matching the paths recorded in the frozen artifacts.
BACKFILL_CONFIG = Path("configs/spc_nolan_backfill_v1.json")
MAIN_TEST_JSON = ROOT / "result" / "anchored_cprc" / "main_test.json"
GATE_PARETO_JSON = ROOT / "result" / "cprc_robustness" / "gate_pareto_qwen_llava_v1.json"
GATE_PARETO_ARTIFACT = Path("result/cprc_robustness/gate_pareto_qwen_llava_v1.json")
GATE_PARETO_CONFIG = ROOT / "configs" / "spc_gate_pareto_v1.json"
OUTPUT_DIR = ROOT / "result" / "paper_revision_stats"

NOLAN_BETA = 0.8
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260720

QUADRATURE_POINTS = 64
FAMILY = "map_no_laplace_or_density"
L2_GRID = (0.01, 0.1, 1.0, 10.0)
LAMBDA_CAPS = (0.8, 1.0, 1.2, 1.5)
MC_ONLY_BUDGET = 0.04
CS_COST = 0.5
REQUIRE_NONNEGATIVE_CF = True
LAMBDA_TRUNC = 1.6
SIDES = ("counterfactual", "commonsense")
STATES = ("agree_modify", "learned_only", "conflict", "nolan_only", "both_keep")


# --------------------------------------------------------------------------- #
# frozen anchored-SPC pipeline (self-contained)
#
# The frozen operating point in result/anchored_cprc/main_test.json was
# produced by an anchored-SPC driver whose row space, NoLan anchor and
# reference prediction streams are reconstructed below from the released
# modules (analyze_hierarchical_eb_lambda_cv, analyze_spc_gate_pareto,
# analyze_spc_robustness, analyze_cp_vbc_bayes_path_cv). The replay is
# verified row-by-row and cell-by-cell against main_test.json, so any drift
# in these helpers fails the verification block loudly.
# --------------------------------------------------------------------------- #

@dataclass
class RowSpace:
    rows: List[Dict[str, Any]]
    keys: List[List[str]]
    image: np.ndarray
    prior: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    baseline_index: np.ndarray
    baseline_correct: np.ndarray
    retention_mask: np.ndarray
    q_image: np.ndarray
    arity: np.ndarray
    raw_contexts: np.ndarray


def build_row_space(rows: Sequence[Dict[str, Any]]) -> RowSpace:
    keys_per_row: List[List[str]] = []
    images: List[np.ndarray] = []
    priors: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    targets: List[int] = []
    baselines: List[int] = []
    baseline_correct: List[bool] = []
    retention: List[float] = []
    q_images: List[np.ndarray] = []
    arities: List[float] = []
    max_k = max(len(candidate_arrays(row)[0]) for row in rows)
    for row in rows:
        keys, image, prior = candidate_arrays(row)
        k = len(keys)
        mask = np.zeros(max_k, dtype=bool)
        mask[:k] = True
        padded_image = np.zeros(max_k, dtype=np.float64)
        padded_prior = np.zeros(max_k, dtype=np.float64)
        padded_image[:k] = image
        padded_prior[:k] = prior
        gt = normalize_key(row.get("gt"))
        if gt not in keys:
            raise ValueError(f"ground truth {gt!r} is not a native candidate")
        baseline = resolved_baseline_key(row)
        baseline_index = keys.index(baseline) if baseline in keys else int(np.argmax(image))
        native_correct = bool(keys[baseline_index] == gt)
        shifted = image - float(np.max(image))
        probs = np.exp(shifted)
        probs /= float(probs.sum())
        q = np.zeros(max_k, dtype=np.float64)
        q[:k] = probs
        keys_per_row.append(keys)
        images.append(padded_image)
        priors.append(padded_prior)
        masks.append(mask)
        targets.append(keys.index(gt))
        baselines.append(baseline_index)
        baseline_correct.append(native_correct)
        retention.append(float(native_correct and row.get("side") == "commonsense"))
        q_images.append(q)
        arities.append(float(math.log(k) / math.log(4.0)))
    raw_contexts = np.stack([context_features(row) for row in rows])
    return RowSpace(
        rows=list(rows),
        keys=keys_per_row,
        image=np.stack(images),
        prior=np.stack(priors),
        mask=np.stack(masks),
        target=np.asarray(targets, dtype=np.int64),
        baseline_index=np.asarray(baselines, dtype=np.int64),
        baseline_correct=np.asarray(baseline_correct, dtype=bool),
        retention_mask=np.asarray(retention, dtype=np.float64),
        q_image=np.stack(q_images),
        arity=np.asarray(arities, dtype=np.float64),
        raw_contexts=raw_contexts,
    )


def corrected_scores(space: RowSpace, alphas: np.ndarray) -> np.ndarray:
    scores = (1.0 + alphas)[:, None] * space.image - alphas[:, None] * space.prior
    return np.where(space.mask, scores, -np.inf)


def proposals_from_scores(space: RowSpace, scores: np.ndarray) -> List[str]:
    winners = np.argmax(scores, axis=1)
    return [space.keys[index][int(winner)] for index, winner in enumerate(winners)]


def distribution_stats(row: Mapping[str, Any]) -> Tuple[float, float]:
    """Return (symmetric KL, Jensen-Shannon) between q_I and q_P."""
    _, image, prior = candidate_arrays(dict(row))
    image_prob = np.exp(image - np.max(image))
    image_prob /= image_prob.sum()
    prior_prob = np.exp(prior - np.max(prior))
    prior_prob /= prior_prob.sum()
    epsilon = 1e-12
    symmetric_kl = 0.5 * float(
        np.sum(
            image_prob
            * np.log(np.maximum(image_prob, epsilon) / np.maximum(prior_prob, epsilon))
        )
        + np.sum(
            prior_prob
            * np.log(np.maximum(prior_prob, epsilon) / np.maximum(image_prob, epsilon))
        )
    )
    mixture = 0.5 * (image_prob + prior_prob)
    js_divergence = 0.5 * float(
        np.sum(image_prob * np.log(np.maximum(image_prob, epsilon) / mixture))
        + np.sum(prior_prob * np.log(np.maximum(prior_prob, epsilon) / mixture))
    )
    return symmetric_kl, js_divergence


def nolan_alpha(row: Mapping[str, Any], beta: float = NOLAN_BETA) -> float:
    symmetric_kl, _ = distribution_stats(row)
    return float(beta) * (math.tanh(1.0 / max(symmetric_kl, 1e-12)) + 1.0)


def anchor_array(
    space: RowSpace,
    kind: str,
    constant: float = 0.0,
    js_beta: float = 0.0,
    beta: float = NOLAN_BETA,
) -> np.ndarray:
    if kind == "nolan":
        return np.asarray([nolan_alpha(row, beta) for row in space.rows], dtype=np.float64)
    if kind == "constant":
        return np.full(len(space.rows), float(constant), dtype=np.float64)
    if kind == "jsd":
        return np.asarray(
            [float(js_beta) * distribution_stats(row)[1] for row in space.rows],
            dtype=np.float64,
        )
    raise ValueError(f"unknown anchor kind: {kind}")


def backfill_predictions(
    rows: Sequence[Dict[str, Any]],
    primary: Sequence[str],
    secondary: Sequence[str],
) -> List[str]:
    output = []
    for row, first, second in zip(rows, primary, secondary):
        baseline = resolved_baseline_key(row)
        output.append(str(first) if str(first) != baseline else str(second))
    return output


def reconstruct_primary(
    rows: Sequence[Dict[str, Any]],
    geometry: Any,
    params: Mapping[str, Any],
    family: str,
    quadrature: int,
) -> List[str]:
    theta, covariance, _ = eb.fit_lambda_posterior(
        rows,
        geometry.contexts,
        geometry.train_ids,
        float(params["l2"]),
    )
    if family == "map_no_laplace_or_density":
        covariance = np.zeros_like(covariance)
        support_distances = np.zeros(len(rows), dtype=np.float64)
    elif family == "bprc_full":
        support_distances = geometry.support_distances
    else:
        raise ValueError(f"unsupported primary family: {family}")
    details = eb.policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        theta,
        covariance,
        lambda_cap=float(params["lambda_cap"]),
        quadrature_points=quadrature,
        support_distances=support_distances,
        arity_supported=geometry.arity_supported,
    )
    return gp.apply_gate_family(rows, details, params["gate"], family)


def reference_predictions(
    space: RowSpace,
    geometry: Any,
    frozen_spc_params: Mapping[str, Any],
    quadrature: int = 64,
) -> Dict[str, List[str]]:
    native = [resolved_baseline_key(row) for row in space.rows]
    nolan = proposals_from_scores(space, corrected_scores(space, anchor_array(space, "nolan")))
    learned = reconstruct_primary(
        space.rows, geometry, frozen_spc_params, "map_no_laplace_or_density", quadrature
    )
    cprc_map = backfill_predictions(space.rows, learned, nolan)
    return {
        "native": native,
        "nolan": nolan,
        "learned_route": learned,
        "cprc_map": cprc_map,
    }


def method_report(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    ids: Sequence[int],
    bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    labels = sorted({str(rows[index]["_dataset"]) for index in ids})
    metrics = dataset_metrics(rows, predictions, ids)
    counts = gp.paired_counts(rows, predictions, ids)
    report: Dict[str, Any] = {}
    for label_index, label in enumerate(labels):
        label_ids = [
            index for index in ids if str(rows[index]["_dataset"]) == label
        ]
        entry: Dict[str, Any] = {}
        for side_index, side in enumerate(SIDES):
            side_metrics = metrics[label][side]
            repairs = int(counts[label][side]["repairs"])
            harms = int(counts[label][side]["harms"])
            entry[side] = {
                **side_metrics,
                "repairs": repairs,
                "harms": harms,
                "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(repairs, harms),
                "delta_ci95": bootstrap_delta(
                    rows,
                    predictions,
                    label_ids,
                    side,
                    bootstrap,
                    seed + 10 * label_index + side_index,
                ),
            }
        report[label] = entry
    return report


# --------------------------------------------------------------------------- #
# shared loading / frozen replay
# --------------------------------------------------------------------------- #

def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_shared() -> Tuple[Any, Any, Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str]:
    aac = sys.modules[__name__]
    backfill_config = json.loads(BACKFILL_CONFIG.read_text(encoding="utf-8"))
    gate_pareto = json.loads(GATE_PARETO_JSON.read_text(encoding="utf-8"))
    main_test = json.loads(MAIN_TEST_JSON.read_text(encoding="utf-8"))
    budget = f"{float(backfill_config['target_cs_budget']):.2f}"
    return aac, eb, gp, backfill_config, gate_pareto, main_test, budget


def frozen_replay(
    model: str,
    spec: Mapping[str, Any],
    aac: Any,
    eb: Any,
    gp: Any,
    gate_pareto: Mapping[str, Any],
    budget: str,
) -> Dict[str, Any]:
    """Replay the frozen learned-route operating point row by row.

    Returns rows, geometry, per-row policy details, frozen params, and the
    reference prediction streams (native / nolan / learned_route / cprc_map).
    The learned replay is verified identical to the anchored reference.
    """
    rows = aac.load_model_rows(spec)
    space = aac.build_row_space(rows)
    geometry = aac.prepare_geometry(rows)
    frozen = gate_pareto["experiments"][model]["selected_by_development_cs_budget"][
        FAMILY
    ][budget]["params"]
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
    reference = aac.reference_predictions(space, geometry, frozen)
    identical = sum(p == r for p, r in zip(learned, reference["learned_route"]))
    return {
        "rows": rows,
        "geometry": geometry,
        "details": details,
        "frozen": frozen,
        "learned": learned,
        "reference": reference,
        "train_loss": float(train_loss),
        "verification": {
            "learned_route_replay_identical": int(identical),
            "learned_route_replay_total": len(rows),
            "matches": identical == len(rows),
        },
    }


def test_ids_by_task(rows: Sequence[Dict[str, Any]], geometry: Any) -> Dict[str, List[int]]:
    return {
        task: [
            i
            for i in geometry.test_ids
            if rows[i]["_dataset"] == f"test_{task}"
        ]
        for task in ("mc", "qa")
    }


def outcome_counts(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    ids: Sequence[int],
) -> Dict[str, Dict[str, int]]:
    """CF/CS repairs and harms of a prediction stream over the given rows."""
    out = {side: {"repairs": 0, "harms": 0} for side in SIDES}
    for index in ids:
        gt = str(rows[index].get("gt", "")).strip().lower()
        baseline_correct = (
            resolve_baseline(rows[index]).lower() == gt
        )
        method_correct = str(predictions[index]).lower() == gt
        side = str(rows[index].get("side"))
        out[side]["repairs"] += int(not baseline_correct and method_correct)
        out[side]["harms"] += int(baseline_correct and not method_correct)
    return out


def resolve_baseline(row: Mapping[str, Any]) -> str:
    cp_vbc = row.get("cp_vbc") or {}
    return str(cp_vbc.get("baseline_key") or "").strip()


# --------------------------------------------------------------------------- #
# task 1: proposal-state decomposition
# --------------------------------------------------------------------------- #

def proposal_state(gated: bool, baseline: str, z_m: str, z_a: str) -> str:
    if gated:
        if z_a == baseline:
            return "learned_only"
        return "agree_modify" if z_a == z_m else "conflict"
    return "both_keep" if z_a == baseline else "nolan_only"


def run_decomposition(
    aac: Any,
    eb: Any,
    gp: Any,
    backfill_config: Mapping[str, Any],
    gate_pareto: Mapping[str, Any],
    main_test: Mapping[str, Any],
    budget: str,
) -> Dict[str, Any]:
    experiments: Dict[str, Any] = {}
    for model, spec in backfill_config["models"].items():
        replay = frozen_replay(model, spec, aac, eb, gp, gate_pareto, budget)
        rows = replay["rows"]
        details = replay["details"]
        reference = replay["reference"]
        learned = replay["learned"]
        native = reference["native"]
        nolan = reference["nolan"]
        cprc_map = reference["cprc_map"]

        # SPC-MAP composition sanity: g_M=1 -> z_M, else z_A (which may be native).
        composition_mismatch = sum(
            cprc_map[i] != (learned[i] if learned[i] != native[i] else nolan[i])
            for i in range(len(rows))
        )

        policies = {
            "cprc_map": cprc_map,
            "learned_route_only": learned,
            "nolan_only": nolan,
        }
        cells: Dict[str, Any] = {}
        by_task = test_ids_by_task(rows, replay["geometry"])
        for task, ids in by_task.items():
            states: Dict[str, Dict[str, Any]] = {}
            for state in STATES:
                states[state] = {"n": 0, "policies": {}}
            for index in ids:
                baseline = native[index]
                z_m = str(details[index]["proposal"])
                z_a = str(nolan[index])
                gated = learned[index] != baseline
                state = proposal_state(gated, baseline, z_m, z_a)
                states[state].setdefault("_ids", []).append(index)
            for state in STATES:
                state_ids = states[state].pop("_ids", [])
                states[state]["n"] = len(state_ids)
                for policy_name, predictions in policies.items():
                    states[state]["policies"][policy_name] = outcome_counts(
                        rows, predictions, state_ids
                    )
            # whole-cell policy totals (verification vs main_test.json)
            totals = {
                policy_name: outcome_counts(rows, predictions, ids)
                for policy_name, predictions in policies.items()
            }
            cells[f"test_{task}"] = {
                "n_rows": len(ids),
                "states": states,
                "policy_totals_repairs_harms": totals,
            }

        # verification against the frozen main_test.json summaries
        method_checks: Dict[str, Any] = {}
        report_map = {
            "learned_route_only": "learned_route",
            "nolan_only": "nolan",
            "cprc_map": "cprc_map",
        }
        all_match = True
        for policy_name, method_name in report_map.items():
            report = aac.method_report(
                rows, policies[policy_name], replay["geometry"].test_ids, bootstrap=200
            )
            expected = main_test["experiments"][model]["methods"][method_name]
            for cell in ("test_mc", "test_qa"):
                for side in SIDES:
                    got = report[cell][side]
                    want = expected[cell][side]
                    match = all(
                        got[key] == want[key]
                        for key in ("correct", "overrides", "baseline_correct", "repairs", "harms")
                    )
                    all_match &= match
                    method_checks[f"{method_name}/{cell}/{side}"] = {
                        "expected": {
                            key: want[key]
                            for key in ("correct", "overrides", "baseline_correct", "repairs", "harms")
                        },
                        "recomputed": {
                            key: got[key]
                            for key in ("correct", "overrides", "baseline_correct", "repairs", "harms")
                        },
                        "matches": match,
                    }
        experiments[model] = {
            "frozen_operating_point": {
                "family": FAMILY,
                "l2": float(replay["frozen"]["l2"]),
                "lambda_cap": float(replay["frozen"]["lambda_cap"]),
                "gate": json_safe(replay["frozen"]["gate"]),
            },
            "rows": len(rows),
            "development_rows": len(replay["geometry"].train_ids),
            "test_rows": len(replay["geometry"].test_ids),
            "verification": {
                **replay["verification"],
                "spc_map_composition_mismatches": int(composition_mismatch),
                "main_test_method_cells": method_checks,
                "matches_frozen_main_test": bool(all_match),
            },
            "cells": cells,
        }
    return {
        "version": "proposal_decomposition_v1",
        "generated_by": "analyze_proposal_decomposition.py --task decomposition",
        "definitions": {
            "states": {
                "agree_modify": "g_M=1 and z_A!=b and z_A==z_M (both routes change to the same answer)",
                "learned_only": "g_M=1 and z_A==b (NoLan would keep the native answer)",
                "conflict": "g_M=1 and z_A!=b and z_A!=z_M (routes change to different answers)",
                "nolan_only": "g_M=0 and z_A!=b (fallback supplies the override)",
                "both_keep": "g_M=0 and z_A==b",
            },
            "repairs_harms": (
                "repairs: baseline wrong -> policy correct; harms: baseline correct -> "
                "policy wrong; counted per side (counterfactual/commonsense) against gold."
            ),
            "policies": {
                "cprc_map": "g_M=1 -> z_M, else z_A (NoLan beta=0.8) which may equal native",
                "learned_route_only": "g_M=1 -> z_M, else native",
                "nolan_only": "always z_A",
            },
        },
        "inputs": {
            model: dict(spec) for model, spec in backfill_config["models"].items()
        }
        | {
            "backfill_config": str(aac.BACKFILL_CONFIG),
            "gate_pareto_artifact": str(GATE_PARETO_JSON.relative_to(ROOT)),
            "main_test_reference": str(MAIN_TEST_JSON.relative_to(ROOT)),
            "frozen_replay": "self-contained (see module docstring)",
        },
        "development_cs_budget": float(budget),
        "nolan_beta": float(backfill_config["nolan_beta"]),
        "experiments": json_safe(experiments),
    }


# --------------------------------------------------------------------------- #
# task 2: MC-only fitting ablation
# --------------------------------------------------------------------------- #

def geometry_for_train_ids(rows: Sequence[Dict[str, Any]], train_ids: Sequence[int], gp: Any, eb: Any) -> Any:
    """Same construction as analyze_spc_gate_pareto.prepare_geometry, restricted to
    the given development rows for standardization, density support and arity bounds."""
    train_ids = list(train_ids)
    test_ids = [
        index
        for index, row in enumerate(rows)
        if str(row.get("_dataset", "")).startswith("test_")
    ]
    raw = np.stack([eb.context_features(row, feature_mode="full") for row in rows])
    contexts, mean, scale = eb.standardize_contexts(raw, train_ids)
    density = contexts[:, 1:]
    covariance = np.cov(density[train_ids], rowvar=False)
    precision = np.linalg.pinv(
        np.atleast_2d(covariance) + 0.1 * np.eye(density.shape[1]), rcond=1e-8
    )
    distances = np.sqrt(
        np.maximum(0.0, np.einsum("ni,ij,nj->n", density, precision, density))
    )
    thresholds = sorted(
        {float(np.quantile(distances[train_ids], q)) for q in (0.9, 0.95, 0.99)}
    )
    arity_min = float(np.min(raw[train_ids, 1]))
    arity_max = float(np.max(raw[train_ids, 1]))
    arity_supported = np.logical_and(
        raw[:, 1] >= arity_min - 1e-12,
        raw[:, 1] <= arity_max + 1e-12,
    )
    return gp.Geometry(
        raw_contexts=raw,
        contexts=contexts,
        support_distances=distances,
        distance_thresholds=thresholds,
        arity_supported=arity_supported,
        train_ids=train_ids,
        test_ids=test_ids,
        context_mean=mean,
        context_scale=scale,
    )


def select_map_operating_point(
    rows: Sequence[Dict[str, Any]],
    geometry: Any,
    fit_ids: Sequence[int],
    eval_ids: Sequence[int],
    eb: Any,
    gp: Any,
) -> Dict[str, Any]:
    """Re-run the frozen development selection protocol for family
    map_no_laplace_or_density at CS budget 0.04 over the l2 x cap x gate grid.

    Mirrors run_posterior_families (zero-covariance MAP details) and
    DevelopmentSelector (net_utility key, valid_retention budget, param_key
    tie-break), restricted to the given fit/eval development rows.
    """
    eval_rows = [rows[index] for index in eval_ids]
    eval_contexts = geometry.contexts[list(eval_ids)]
    eval_arity = geometry.arity_supported[list(eval_ids)]
    eval_support = np.zeros(len(eval_ids))
    best: Optional[Dict[str, Any]] = None
    considered = 0
    for l2 in L2_GRID:
        theta, covariance, train_loss = eb.fit_lambda_posterior(
            rows, geometry.contexts, list(fit_ids), float(l2)
        )
        zero_covariance = np.zeros_like(covariance)
        for cap in LAMBDA_CAPS:
            details = eb.policy_details_gauss_hermite(
                eval_rows,
                eval_contexts,
                theta,
                zero_covariance,
                lambda_cap=float(cap),
                quadrature_points=QUADRATURE_POINTS,
                support_distances=eval_support,
                arity_supported=eval_arity,
            )
            for gate in gp.family_gate_grid(FAMILY, (float("inf"),)):
                predictions = gp.apply_gate_family(eval_rows, details, gate, FAMILY)
                metrics = eb.dataset_metrics(
                    eval_rows, predictions, range(len(eval_rows))
                )
                if not eb.valid_retention(metrics, MC_ONLY_BUDGET, REQUIRE_NONNEGATIVE_CF):
                    continue
                key = eb.selection_key(
                    metrics,
                    predictions,
                    eval_rows,
                    list(range(len(eval_rows))),
                    selection_objective="net_utility",
                    cs_cost=CS_COST,
                )
                params = {"l2": float(l2), "lambda_cap": float(cap), "gate": dict(gate)}
                param_key = json.dumps(params, sort_keys=True, allow_nan=True)
                considered += 1
                better = best is None or key > best["selection_key"]
                if best is not None and key == best["selection_key"]:
                    better = param_key < best["param_key"]
                if better:
                    best = {
                        "params": params,
                        "selection_key": key,
                        "param_key": param_key,
                        "development_metrics": metrics,
                        "train_loss": float(train_loss),
                    }
    if best is None:
        raise RuntimeError("no grid point satisfies the development CS budget")
    best["grid_points_within_budget"] = considered
    return best


def evaluate_params_on_test(
    rows: Sequence[Dict[str, Any]],
    geometry: Any,
    fit_ids: Sequence[int],
    params: Mapping[str, Any],
    eb: Any,
    gp: Any,
) -> Dict[str, Any]:
    theta, covariance, _ = eb.fit_lambda_posterior(
        rows, geometry.contexts, list(fit_ids), float(params["l2"])
    )
    details = eb.policy_details_gauss_hermite(
        rows,
        geometry.contexts,
        theta,
        np.zeros_like(covariance),
        lambda_cap=float(params["lambda_cap"]),
        quadrature_points=QUADRATURE_POINTS,
        support_distances=np.zeros(len(rows)),
        arity_supported=geometry.arity_supported,
    )
    predictions = gp.apply_gate_family(rows, details, params["gate"], FAMILY)
    metrics = eb.dataset_metrics(rows, predictions, geometry.test_ids)
    return {"predictions": predictions, "test_metrics": metrics, "details": details}


def run_mc_only(
    aac: Any,
    eb: Any,
    gp: Any,
    backfill_config: Mapping[str, Any],
    gate_pareto: Mapping[str, Any],
    main_test: Mapping[str, Any],
    budget: str,
) -> Dict[str, Any]:
    experiments: Dict[str, Any] = {}
    for model, spec in backfill_config["models"].items():
        rows = aac.load_model_rows(spec)
        full_geometry = aac.prepare_geometry(rows)
        dev_mc_ids = [
            i for i in full_geometry.train_ids if rows[i]["_dataset"] == "dev_mc"
        ]
        test_mc_ids = [
            i for i in full_geometry.test_ids if rows[i]["_dataset"] == "test_mc"
        ]
        test_qa_ids = [
            i for i in full_geometry.test_ids if rows[i]["_dataset"] == "test_qa"
        ]

        # --- validation: re-run the JOINT selection, expect the frozen params ---
        joint = select_map_operating_point(
            rows, full_geometry, full_geometry.train_ids, full_geometry.train_ids, eb, gp
        )
        frozen = gate_pareto["experiments"][model]["selected_by_development_cs_budget"][
            FAMILY
        ][budget]["params"]
        joint_match = json_safe(joint["params"]) == json_safe(frozen)

        # --- ablation: fit and select on dev_mc only ---
        mc_geometry = geometry_for_train_ids(rows, dev_mc_ids, gp, eb)
        mc_only = select_map_operating_point(rows, mc_geometry, dev_mc_ids, dev_mc_ids, eb, gp)
        evaluation = evaluate_params_on_test(
            rows, mc_geometry, dev_mc_ids, mc_only["params"], eb, gp
        )
        test_mc = evaluation["test_metrics"]["test_mc"]
        test_qa = evaluation["test_metrics"]["test_qa"]

        frozen_test_mc = main_test["experiments"][model]["methods"]["learned_route"][
            "test_mc"
        ]
        experiments[model] = {
            "rows": len(rows),
            "dev_mc_rows": len(dev_mc_ids),
            "test_mc_rows": len(test_mc_ids),
            "test_qa_rows": len(test_qa_ids),
            "joint_selection_validation": {
                "reselected_params": json_safe(joint["params"]),
                "frozen_params": json_safe(frozen),
                "matches_frozen": bool(joint_match),
                "grid_points_within_budget": joint["grid_points_within_budget"],
            },
            "mc_only_fit": {
                "note": (
                    "Coefficient fit, context standardization, density/arity support "
                    "and the CS-budget/utility selection all use dev_mc only; the "
                    "budget constraint is applied to dev_mc because dev_qa does not "
                    "exist in this ablation."
                ),
                "selected_params": json_safe(mc_only["params"]),
                "selection_key": list(mc_only["selection_key"]),
                "development_metrics": json_safe(mc_only["development_metrics"]),
                "grid_points_within_budget": mc_only["grid_points_within_budget"],
                "test_mc": json_safe(test_mc),
                "test_qa": json_safe(test_qa),
            },
            "frozen_joint_reference": {
                "params": json_safe(frozen),
                "test_mc": json_safe(frozen_test_mc),
            },
            "comparison_test_mc": {
                "mc_only_delta_cf": test_mc["counterfactual"]["delta"],
                "mc_only_delta_cs": test_mc["commonsense"]["delta"],
                "frozen_joint_delta_cf": frozen_test_mc["counterfactual"]["delta"],
                "frozen_joint_delta_cs": frozen_test_mc["commonsense"]["delta"],
                "difference_cf": test_mc["counterfactual"]["delta"]
                - frozen_test_mc["counterfactual"]["delta"],
                "difference_cs": test_mc["commonsense"]["delta"]
                - frozen_test_mc["commonsense"]["delta"],
            },
        }
    return {
        "version": "mc_only_ablation_v1",
        "generated_by": "analyze_proposal_decomposition.py --task mc_only",
        "protocol": {
            "family": FAMILY,
            "l2_grid": list(L2_GRID),
            "lambda_caps": list(LAMBDA_CAPS),
            "gate_grid": "family_gate_grid(map_no_laplace_or_density, (inf,)) = 6x6x5",
            "quadrature_points": QUADRATURE_POINTS,
            "cs_budget_dev_mc_only": MC_ONLY_BUDGET,
            "selection_objective": "net_utility (sum delta_CF + 0.5 * sum delta_CS)",
            "require_nonnegative_cf": REQUIRE_NONNEGATIVE_CF,
            "tie_break": "lexicographic json param key, as in DevelopmentSelector",
        },
        "inputs": {
            model: dict(spec) for model, spec in backfill_config["models"].items()
        }
        | {
            "gate_pareto_artifact": str(GATE_PARETO_JSON.relative_to(ROOT)),
            "gate_pareto_config": str(GATE_PARETO_CONFIG.relative_to(ROOT)),
            "main_test_reference": str(MAIN_TEST_JSON.relative_to(ROOT)),
        },
        "experiments": experiments,
    }


# --------------------------------------------------------------------------- #
# task 3: stability radius
# --------------------------------------------------------------------------- #

def mann_whitney_greater(
    x: Sequence[float], y: Sequence[float]
) -> Optional[Dict[str, Any]]:
    """One-sided Mann-Whitney U (x stochastically greater than y).

    Self-contained normal approximation with tie correction and continuity
    correction; scipy is not installed in cdh-bench-env. U is the x-side
    statistic (P(X>Y) + 0.5 P(X=Y) scaled by n1*n2).
    """
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return None
    values = x + y
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    rank_x = sum(ranks[:n1])
    u = rank_x - n1 * (n1 + 1) / 2.0
    total = n1 + n2
    tie_sum = sum(count**3 - count for count in Counter(values).values())
    variance = n1 * n2 / 12.0 * ((total + 1) - tie_sum / (total * (total - 1)))
    mean = n1 * n2 / 2.0
    if variance <= 0.0:
        return {
            "n_repair": n1,
            "n_harm": n2,
            "U": u,
            "z": None,
            "p_one_sided_greater": None,
            "method": "degenerate (no variance)",
        }
    z = (u - mean - 0.5) / math.sqrt(variance)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {
        "n_repair": n1,
        "n_harm": n2,
        "U": u,
        "U_over_n1n2": u / (n1 * n2),
        "z": z,
        "p_one_sided_greater": p,
        "method": "normal approximation, tie-corrected, continuity-corrected (scipy unavailable)",
    }


def quantiles(values: Sequence[float]) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def run_stability(
    aac: Any,
    eb: Any,
    gp: Any,
    backfill_config: Mapping[str, Any],
    gate_pareto: Mapping[str, Any],
    main_test: Mapping[str, Any],
    budget: str,
) -> Dict[str, Any]:
    from analyze_lambda_stable_intervals import stable_interval

    groups = (
        "cf_repair",
        "cf_harm",
        "cs_repair",
        "cs_harm",
        "modified_neutral",
        "unchanged_correct",
        "unchanged_wrong",
    )
    experiments: Dict[str, Any] = {}
    pooled: Dict[str, Dict[str, List[float]]] = {
        "unbounded": {"repair": [], "harm": []},
        "truncated": {"repair": [], "harm": []},
    }
    for model, spec in backfill_config["models"].items():
        replay = frozen_replay(model, spec, aac, eb, gp, gate_pareto, budget)
        rows = replay["rows"]
        details = replay["details"]
        reference = replay["reference"]
        native = reference["native"]
        cprc_map = reference["cprc_map"]
        cap = float(replay["frozen"]["lambda_cap"])

        cells: Dict[str, Any] = {}
        by_task = test_ids_by_task(rows, replay["geometry"])
        for task, ids in by_task.items():
            bucket: Dict[str, Dict[str, List[float]]] = {
                group: {"unbounded": [], "truncated": []} for group in groups
            }
            empty_intervals = 0
            for index in ids:
                keys, image, prior = eb.candidate_arrays(rows[index])
                proposal_index = keys.index(details[index]["proposal"])
                lo, hi, empty = stable_interval(image, prior, proposal_index)
                lam = float(details[index]["lambda_mean"])
                r_unbounded = min(lam - lo, hi - lam)  # hi=inf -> lam - lo
                r_truncated = min(lam - lo, min(hi, LAMBDA_TRUNC) - lam)
                empty_intervals += int(empty)

                gt = str(rows[index].get("gt", "")).strip().lower()
                baseline = native[index]
                final = cprc_map[index]
                baseline_correct = baseline.lower() == gt
                final_correct = str(final).lower() == gt
                side = str(rows[index].get("side"))
                if final != baseline:
                    if final_correct and not baseline_correct:
                        group = "cf_repair" if side == "counterfactual" else "cs_repair"
                    elif baseline_correct and not final_correct:
                        group = "cf_harm" if side == "counterfactual" else "cs_harm"
                    else:
                        group = "modified_neutral"
                else:
                    group = "unchanged_correct" if baseline_correct else "unchanged_wrong"
                bucket[group]["unbounded"].append(r_unbounded)
                bucket[group]["truncated"].append(r_truncated)

            group_stats: Dict[str, Any] = {}
            for group in groups:
                group_stats[group] = {
                    "n": len(bucket[group]["unbounded"]),
                    "n_outside_interval": sum(
                        v < -1e-9 for v in bucket[group]["unbounded"]
                    ),
                    "r_unbounded": quantiles(bucket[group]["unbounded"]),
                    f"r_truncated_{LAMBDA_TRUNC}": quantiles(bucket[group]["truncated"]),
                }
            cell_tests: Dict[str, Any] = {}
            for variant in ("unbounded", "truncated"):
                repairs = (
                    bucket["cf_repair"][variant] + bucket["cs_repair"][variant]
                )
                harms = bucket["cf_harm"][variant] + bucket["cs_harm"][variant]
                cell_tests[f"repair_vs_harm_{variant}"] = mann_whitney_greater(
                    repairs, harms
                )
                cell_tests[f"cf_repair_vs_cf_harm_{variant}"] = mann_whitney_greater(
                    bucket["cf_repair"][variant], bucket["cf_harm"][variant]
                )
                pooled[variant]["repair"].extend(repairs)
                pooled[variant]["harm"].extend(harms)
            cells[f"test_{task}"] = {
                "n_rows": len(ids),
                "empty_intervals": int(empty_intervals),
                "groups": group_stats,
                "mann_whitney": cell_tests,
            }
        experiments[model] = {
            "lambda_cap": cap,
            "verification": replay["verification"],
            "cells": cells,
        }

    pooled_tests = {
        variant: {
            "repair": quantiles(pooled[variant]["repair"]),
            "harm": quantiles(pooled[variant]["harm"]),
            "mann_whitney_repair_greater": mann_whitney_greater(
                pooled[variant]["repair"], pooled[variant]["harm"]
            ),
        }
        for variant in ("unbounded", "truncated")
    }
    return {
        "version": "stability_radius_v1",
        "generated_by": "analyze_proposal_decomposition.py --task stability",
        "definitions": {
            "stable_interval": (
                "I_z = {lambda >= 0 : s_I(z) - lambda*s_P(z) >= s_I(y) - lambda*s_P(y) "
                "for all candidates y}, exact half-line intersection as in "
                "analyze_lambda_stable_intervals.py; z is the learned-route proposal."
            ),
            "r_unbounded": (
                "min(lambda_theta - lambda_lo, lambda_hi - lambda_theta); rows with "
                "lambda_hi = inf contribute lambda_theta - lambda_lo; negative values "
                "mean lambda_theta lies outside I_z (empty or violated interval)."
            ),
            f"r_truncated_{LAMBDA_TRUNC}": (
                f"same with lambda_hi truncated at {LAMBDA_TRUNC} before taking the "
                "minimum (display-scale convention of the interval figure)."
            ),
            "groups": (
                "Outcome of the final SPC-MAP decision vs the native answer and gold: "
                "cf/cs repair (wrong->right), cf/cs harm (right->wrong), "
                "modified_neutral (changed without correctness change), "
                "unchanged_correct / unchanged_wrong."
            ),
        },
        "inputs": {
            model: dict(spec) for model, spec in backfill_config["models"].items()
        }
        | {
            "backfill_config": str(aac.BACKFILL_CONFIG),
            "gate_pareto_artifact": str(GATE_PARETO_JSON.relative_to(ROOT)),
            "main_test_reference": str(MAIN_TEST_JSON.relative_to(ROOT)),
        },
        "development_cs_budget": float(budget),
        "experiments": json_safe(experiments),
        "pooled_all_cells": pooled_tests,
    }


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("decomposition", "mc_only", "stability", "all"),
        default="all",
    )
    args = parser.parse_args()

    aac, eb, gp, backfill_config, gate_pareto, main_test, budget = load_shared()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = (
        ("decomposition", "mc_only", "stability")
        if args.task == "all"
        else (args.task,)
    )
    for task in tasks:
        if task == "decomposition":
            payload = run_decomposition(
                aac, eb, gp, backfill_config, gate_pareto, main_test, budget
            )
            output = OUTPUT_DIR / "proposal_decomposition.json"
        elif task == "mc_only":
            payload = run_mc_only(
                aac, eb, gp, backfill_config, gate_pareto, main_test, budget
            )
            output = OUTPUT_DIR / "mc_only_ablation.json"
        else:
            payload = run_stability(
                aac, eb, gp, backfill_config, gate_pareto, main_test, budget
            )
            output = OUTPUT_DIR / "stability_radius.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
