#!/usr/bin/env python3
"""Per-row stable intervals I_z of the frozen EB-SPC learned route.

For every blind-test row this script replays the frozen learned-route operating
point (dev70 joint MC+QA fit, l2=0.01, cap=1.0 for Qwen3-VL-32B-Instruct and
cap=0.8 for LLaVA-1.6-34B, development CS budget 0.04), recovers the per-row
context-conditioned coefficient lambda_theta(o) and the learned-route proposal
z, and solves the linear inequalities of the paper's stable-interval definition

    I_z = { lambda >= 0 : s_I(z) - lambda * s_P(z) >= s_I(y) - lambda * s_P(y)  for all y }

exactly, using the same mean-centered candidate scores as candidate_arrays in
analyze_hierarchical_eb_lambda_cv.py. It then reports, per model x format cell:

  (a) fraction of rows with lambda_theta inside I_z (tolerance 1e-9);
  (b) distribution of the interval width (median, IQR, infinite-width count);
  (c) position of the cap relative to the interval upper bound, and, among rows
      whose lambda_theta touches the cap, the fraction with cap inside I_z;
  (d) the same in-interval fraction for the dev-fitted constant lambda*
      (fixed_lambda_same_gate family), as the constant-coefficient control.

The replay is verified row-by-row against reference_predictions of the frozen
anchored_spc pipeline and cell-by-cell against the learned-route summary of
result/anchored_cprc/main_test.json; the verification is stored in the output.

Outputs:
  result/paper_revision_stats/lambda_intervals.json
  paper/aaai27/figures/lambda_stable_intervals.pdf (+ .png preview)

Re-run:  cdh-bench-env/bin/python analyze_lambda_stable_intervals.py
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

import analyze_proposal_decomposition as apd

ROOT = Path(__file__).resolve().parent
MAIN_TEST_JSON = ROOT / "result" / "anchored_cprc" / "main_test.json"
REVISION_STATS_JSON = ROOT / "result" / "paper_revision_stats" / "stats.json"
OUTPUT_JSON = ROOT / "result" / "paper_revision_stats" / "lambda_intervals.json"
FIGURE_PDF = ROOT / "paper" / "aaai27" / "figures" / "lambda_stable_intervals.pdf"

TOL = 1e-9
QUADRATURE_POINTS = 64
LAMBDA_DISPLAY_MAX = 1.6

# Paper-figure style, matched to paper/aaai27/make_paper_figures.py.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#6B7280"
INK = "#1F2937"
GRID = "#E5E7EB"


def load_anchored_module() -> Any:
    """Return the self-contained frozen-replay namespace (analyze_proposal_decomposition)."""
    return apd


def setup_figure_style() -> None:
    for font_dir in (
        Path("/tmp/tinytex-aaai/.TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"),
        Path("/usr/share/fonts/opentype/texgyre"),
        Path("/usr/local/share/fonts"),
    ):
        if not font_dir.exists():
            continue
        for font_path in font_dir.glob("texgyretermes-*.otf"):
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def stable_interval(
    image: np.ndarray, prior: np.ndarray, proposal_index: int
) -> Tuple[float, float, bool]:
    """Exact I_z = [lambda_lo, lambda_hi] for the proposal under centered scores.

    For each rival y the inequality s_I(z) - s_I(y) >= lambda * (s_P(z) - s_P(y))
    contributes one half-line bound; lambda_hi may be inf. Returns (lo, hi, empty).
    """
    z = proposal_index
    lower, upper = 0.0, float("inf")
    for y in range(len(image)):
        if y == z:
            continue
        image_delta = float(image[z] - image[y])
        prior_delta = float(prior[z] - prior[y])
        if abs(prior_delta) < 1e-12:
            if image_delta < 0.0:
                return 0.0, 0.0, True
            continue
        boundary = image_delta / prior_delta
        if prior_delta > 0.0:
            upper = min(upper, boundary)
        else:
            lower = max(lower, boundary)
    lower = max(0.0, lower)
    if upper < lower:
        return lower, upper, True
    return lower, upper, False


def quantile_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
    }


def fit_constant_lambda(
    rows: Sequence[Dict[str, Any]],
    train_ids: Sequence[int],
    l2: float,
    cap: float,
    fit_lambda_posterior: Any,
    sigmoid_softplus: Any,
) -> Dict[str, float]:
    """Refit the intercept-only (constant) coefficient, as in fixed_lambda_same_gate."""
    constant_contexts = np.ones((len(rows), 1), dtype=np.float64)
    theta, _, train_loss = fit_lambda_posterior(rows, constant_contexts, train_ids, l2)
    uncapped = float(sigmoid_softplus(constant_contexts @ theta)[0])
    return {
        "l2": float(l2),
        "cap": float(cap),
        "lambda_uncapped": uncapped,
        "lambda": min(uncapped, float(cap)),
        "train_loss": float(train_loss),
        "fit_rows": len(list(train_ids)),
    }


def analyze_model(
    model: str,
    spec: Mapping[str, Any],
    aac: Any,
    gate_pareto: Mapping[str, Any],
    main_test: Mapping[str, Any],
    budget: str,
    eb: Any,
    gp: Any,
) -> Dict[str, Any]:
    rows = aac.load_model_rows(spec)
    space = aac.build_row_space(rows)
    geometry = aac.prepare_geometry(rows)

    frozen = gate_pareto["experiments"][model]["selected_by_development_cs_budget"][
        "map_no_laplace_or_density"
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
    predictions = gp.apply_gate_family(rows, details, frozen["gate"], "map_no_laplace_or_density")

    # --- verification against the frozen pipeline ---
    reference = aac.reference_predictions(space, geometry, frozen)
    identical = sum(p == r for p, r in zip(predictions, reference["learned_route"]))
    report = aac.method_report(rows, predictions, geometry.test_ids, bootstrap=200)
    expected = main_test["experiments"][model]["methods"]["learned_route"]
    cells = {}
    all_match = identical == len(rows)
    for cell in ("test_mc", "test_qa"):
        for side in ("counterfactual", "commonsense"):
            got = report[cell][side]
            want = expected[cell][side]
            match = (
                got["correct"] == want["correct"]
                and got["overrides"] == want["overrides"]
                and got["baseline_correct"] == want["baseline_correct"]
            )
            all_match &= match
            cells[f"{cell}/{side}"] = {
                "expected_correct": want["correct"],
                "recomputed_correct": got["correct"],
                "expected_overrides": want["overrides"],
                "recomputed_overrides": got["overrides"],
                "matches": match,
            }
    verification = {
        "reference_replay_row_predictions": {
            "identical": int(identical),
            "total": len(rows),
            "matches": identical == len(rows),
        },
        "main_test_learned_route_cells": cells,
        "matches_frozen_main_test": bool(all_match),
    }

    # --- constant-lambda control (fixed_lambda_same_gate frozen selection) ---
    constant_frozen = gate_pareto["experiments"][model][
        "selected_by_development_cs_budget"
    ]["fixed_lambda_same_gate"][budget]["params"]
    constant_l2 = float(constant_frozen["l2"])
    constant_cap = float(constant_frozen["lambda_cap"])
    dev_mc_ids = [i for i in geometry.train_ids if rows[i]["_dataset"] == "dev_mc"]
    dev_qa_ids = [i for i in geometry.train_ids if rows[i]["_dataset"] == "dev_qa"]
    constant = {
        "joint_dev70": fit_constant_lambda(
            rows, geometry.train_ids, constant_l2, constant_cap,
            eb.fit_lambda_posterior, eb.sigmoid_softplus,
        ),
        "dev_mc_only": fit_constant_lambda(
            rows, dev_mc_ids, constant_l2, constant_cap,
            eb.fit_lambda_posterior, eb.sigmoid_softplus,
        ),
        "dev_qa_only": fit_constant_lambda(
            rows, dev_qa_ids, constant_l2, constant_cap,
            eb.fit_lambda_posterior, eb.sigmoid_softplus,
        ),
    }
    lambda_star = constant["joint_dev70"]["lambda"]

    # --- per-row intervals and cell statistics ---
    cell_results: Dict[str, Any] = {}
    row_records: Dict[str, List[Dict[str, Any]]] = {}
    for task in ("mc", "qa"):
        test_ids = [
            i for i in geometry.test_ids if rows[i]["_dataset"] == f"test_{task}"
        ]
        records = []
        for index in test_ids:
            keys, image, prior = eb.candidate_arrays(rows[index])
            proposal = details[index]["proposal"]
            proposal_index = keys.index(proposal)
            lo, hi, empty = stable_interval(image, prior, proposal_index)
            lambda_theta = float(details[index]["lambda_mean"])
            records.append(
                {
                    "index": index,
                    "side": str(rows[index].get("side")),
                    "proposal": proposal,
                    "lambda_theta": lambda_theta,
                    "lambda_theta_uncapped": float(details[index]["lambda_raw_mean"]),
                    "lambda_lo": lo,
                    "lambda_hi": hi,
                    "interval_empty": empty,
                    "width": (hi - lo) if not empty else None,
                    "lambda_theta_inside": (not empty)
                    and lo - TOL <= lambda_theta <= hi + TOL,
                    "lambda_star_inside": (not empty) and lo - TOL <= lambda_star <= hi + TOL,
                    "cap_le_lambda_hi": (not empty) and cap <= hi + TOL,
                    "touches_cap": lambda_theta >= cap - TOL,
                    "cap_inside_interval": (not empty) and lo - TOL <= cap <= hi + TOL,
                }
            )
        row_records[task] = records
        n = len(records)
        finite_widths = [
            r["width"] for r in records if r["width"] is not None and math.isfinite(r["width"])
        ]
        touch = [r for r in records if r["touches_cap"]]
        per_side = {}
        for side in ("counterfactual", "commonsense"):
            side_rows = [r for r in records if r["side"] == side]
            per_side[side] = {
                "n": len(side_rows),
                "lambda_theta_inside": sum(r["lambda_theta_inside"] for r in side_rows),
                "lambda_star_inside": sum(r["lambda_star_inside"] for r in side_rows),
            }
        cell_results[f"test_{task}"] = {
            "n_rows": n,
            "a_lambda_theta_inside_interval": {
                "count": sum(r["lambda_theta_inside"] for r in records),
                "fraction": sum(r["lambda_theta_inside"] for r in records) / n,
                "tolerance": TOL,
                "empty_intervals": sum(r["interval_empty"] for r in records),
                "per_side": per_side,
            },
            "b_interval_width": {
                **quantile_summary(finite_widths),
                "finite_count": len(finite_widths),
                "infinite_count": n - len(finite_widths) - sum(r["interval_empty"] for r in records),
            },
            "c_cap_vs_interval": {
                "cap": cap,
                "cap_le_lambda_hi_count": sum(r["cap_le_lambda_hi"] for r in records),
                "cap_le_lambda_hi_fraction": sum(r["cap_le_lambda_hi"] for r in records) / n,
                "lambda_theta_touching_cap": len(touch),
                "touching_cap_with_cap_inside_interval": sum(
                    r["cap_inside_interval"] for r in touch
                ),
            },
            "d_constant_lambda_control": {
                "lambda_star": lambda_star,
                "inside_interval_count": sum(r["lambda_star_inside"] for r in records),
                "inside_interval_fraction": sum(r["lambda_star_inside"] for r in records) / n,
            },
        }

    return {
        "rows": len(rows),
        "development_rows": len(geometry.train_ids),
        "test_rows": len(geometry.test_ids),
        "frozen_operating_point": {
            "family": "map_no_laplace_or_density",
            "l2": l2,
            "lambda_cap": cap,
            "gate": {k: (v if math.isfinite(float(v)) else "inf") for k, v in frozen["gate"].items()},
            "quadrature_points": QUADRATURE_POINTS,
            "train_loss": float(train_loss),
        },
        "constant_lambda_fit": constant,
        "verification": verification,
        "cells": cell_results,
        "_records": row_records,
    }


def make_figure(results: Mapping[str, Any], lambda_stars: Mapping[str, float]) -> None:
    setup_figure_style()
    panels = [
        ("Qwen3-VL-32B-Instruct", "mc", "Qwen3-VL-32B (MC)"),
        ("LLaVA-1.6-34B", "mc", "LLaVA-1.6-34B (MC)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.4), sharey=False)
    for ax, (model, task, title) in zip(axes, panels):
        records = sorted(results[model]["_records"][task], key=lambda r: (r["lambda_lo"], r["lambda_hi"]))
        cap = results[model]["frozen_operating_point"]["lambda_cap"]
        for y, record in enumerate(records):
            lo = record["lambda_lo"]
            hi = record["lambda_hi"]
            truncated = hi > LAMBDA_DISPLAY_MAX
            hi_draw = min(hi, LAMBDA_DISPLAY_MAX)
            ax.plot(
                [lo, hi_draw],
                [y, y],
                color=GRAY,
                alpha=0.18,
                linewidth=0.35,
                solid_capstyle="round",
                zorder=1,
            )
            if truncated:
                ax.plot(
                    [hi_draw],
                    [y],
                    marker=">",
                    markersize=1.4,
                    color=GRAY,
                    alpha=0.45,
                    zorder=1,
                )
        inside_x = [r["lambda_theta"] for r in records if r["lambda_theta_inside"]]
        inside_y = [y for y, r in enumerate(records) if r["lambda_theta_inside"]]
        outside_x = [r["lambda_theta"] for r in records if not r["lambda_theta_inside"]]
        outside_y = [y for y, r in enumerate(records) if not r["lambda_theta_inside"]]
        ax.plot(inside_x, inside_y, ".", color=BLUE, markersize=1.8, zorder=3)
        if outside_x:
            ax.plot(outside_x, outside_y, ".", color=ORANGE, markersize=2.2, zorder=4)
        ax.axvline(cap, color=ORANGE, linestyle="--", linewidth=0.8, zorder=2)
        ax.axvline(lambda_stars[model], color=INK, linestyle=":", linewidth=0.8, zorder=2)
        ax.set_xlim(0.0, LAMBDA_DISPLAY_MAX)
        ax.set_ylim(-8, len(records) + 8)
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda$")
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#9CA3AF")
            spine.set_linewidth(0.6)
        ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test rows (sorted by $\\lambda_{\\mathrm{lo}}$)")
    handles = [
        Line2D([0], [0], color=GRAY, alpha=0.6, linewidth=1.0, label=r"stable interval $I_z$"),
        Line2D([0], [0], marker=".", linestyle="", color=BLUE, markersize=3,
               label=r"$\lambda_\theta(o)$ (inside $I_z$)"),
        Line2D([0], [0], marker=".", linestyle="", color=ORANGE, markersize=3,
               label=r"$\lambda_\theta(o)$ (outside)"),
        Line2D([0], [0], color=ORANGE, linestyle="--", linewidth=0.8, label=r"cap $\lambda_{\max}$"),
        Line2D([0], [0], color=INK, linestyle=":", linewidth=0.8, label=r"constant $\lambda^\ast$"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.9,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.tight_layout(pad=0.4, rect=(0.0, 0.055, 1.0, 1.0))
    FIGURE_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(FIGURE_PDF.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main() -> int:
    import analyze_hierarchical_eb_lambda_cv as eb
    import analyze_spc_gate_pareto as gp

    aac = load_anchored_module()
    backfill_config = json.loads(aac.BACKFILL_CONFIG.read_text(encoding="utf-8"))
    gate_pareto = json.loads(aac.GATE_PARETO_ARTIFACT.read_text(encoding="utf-8"))
    main_test = json.loads(MAIN_TEST_JSON.read_text(encoding="utf-8"))
    budget = f"{float(backfill_config['target_cs_budget']):.2f}"

    results: Dict[str, Any] = {}
    for model, spec in backfill_config["models"].items():
        results[model] = analyze_model(
            model, spec, aac, gate_pareto, main_test, budget, eb, gp
        )
        ver = results[model]["verification"]
        print(
            f"{model}: replay identical {ver['reference_replay_row_predictions']['identical']}"
            f"/{ver['reference_replay_row_predictions']['total']},"
            f" main_test cells match: {ver['matches_frozen_main_test']}"
        )

    # Cross-check the constant-lambda refit against the frozen revision statistics.
    revision_stats = json.loads(REVISION_STATS_JSON.read_text(encoding="utf-8"))
    item4 = revision_stats["item4_constant_lambda"]
    qwen_joint = results["Qwen3-VL-32B-Instruct"]["constant_lambda_fit"]["joint_dev70"]
    qwen_mc_only = results["Qwen3-VL-32B-Instruct"]["constant_lambda_fit"]["dev_mc_only"]
    constant_crosscheck = {
        "reference": "result/paper_revision_stats/stats.json item4_constant_lambda",
        "qwen_joint_lambda_uncapped": {
            "expected": item4["lambda_star_constant_uncapped"],
            "recomputed": qwen_joint["lambda_uncapped"],
            "matches": abs(item4["lambda_star_constant_uncapped"] - qwen_joint["lambda_uncapped"]) < 1e-9,
        },
        "qwen_dev_mc_only_lambda_uncapped": {
            "expected": item4["lambda_star_constant_dev_mc_only_uncapped"],
            "recomputed": qwen_mc_only["lambda_uncapped"],
            "matches": abs(item4["lambda_star_constant_dev_mc_only_uncapped"] - qwen_mc_only["lambda_uncapped"]) < 1e-9,
        },
    }

    lambda_stars = {
        model: res["constant_lambda_fit"]["joint_dev70"]["lambda"]
        for model, res in results.items()
    }
    make_figure(results, lambda_stars)

    def json_safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return "inf" if value > 0 else "-inf"
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items() if k != "_records"}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return value

    payload = {
        "version": "lambda_stable_intervals_v1",
        "generated_by": "analyze_lambda_stable_intervals.py",
        "definition": (
            "I_z = {lambda >= 0 : s_I(z) - lambda*s_P(z) >= s_I(y) - lambda*s_P(y) for all "
            "candidates y}, with mean-centered candidate scores as in candidate_arrays of "
            "analyze_hierarchical_eb_lambda_cv.py; z is the learned-route proposal and "
            "lambda_theta the per-row MAP coefficient of the frozen operating point."
        ),
        "inputs": {
            "qwen_dev70": backfill_config["models"]["Qwen3-VL-32B-Instruct"]["dev_mc"],
            "qwen_test": backfill_config["models"]["Qwen3-VL-32B-Instruct"]["test"],
            "llava_dev70": backfill_config["models"]["LLaVA-1.6-34B"]["dev_mc"],
            "llava_test": backfill_config["models"]["LLaVA-1.6-34B"]["test"],
            "backfill_config": str(aac.BACKFILL_CONFIG),
            "gate_pareto_artifact": str(aac.GATE_PARETO_ARTIFACT),
            "main_test_reference": str(MAIN_TEST_JSON.relative_to(ROOT)),
            "frozen_replay": "analyze_proposal_decomposition.py (self-contained)",
        },
        "development_cs_budget": float(budget),
        "quadrature_points": QUADRATURE_POINTS,
        "constant_lambda_crosscheck": constant_crosscheck,
        "experiments": json_safe(results),
        "figure": str(FIGURE_PDF.relative_to(ROOT)),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"wrote {FIGURE_PDF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
