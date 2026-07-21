"""Redraw paper Figure 2 (evidence chain) as a two-panel full-width figure.

Panel (a): prior-direction controls on Qwen3-VL-32B MC (frozen High-Repair
attribution point). Data: result/cprc_robustness/prior_controls_qwen32b_v1.json
Panel (b): CF-repair / CS-retention frontier traced by the learned MAP route
on both models, with the primary operating point (star) and the matched
constant-coefficient control (open diamond).
Data: result/cprc_robustness/gate_pareto_qwen_llava_v1.json

Outputs: figures/evidence_chain_v2.pdf / .png
Style follows make_paper_figures.py (TeX Gyre Termes, 0.7 pt axes, Okabe-Ito
palette) with cleaner, larger panels: tick labels >= 8 pt, no chartjunk.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6B7280"
INK = "#1F2937"
GRID = "#E5E7EB"


def setup() -> None:
    font_dirs = (
        Path("/tmp/tinytex-aaai/.TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre"),
        Path("/usr/share/fonts/opentype/texgyre"),
        Path("/usr/local/share/fonts"),
    )
    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for font_path in font_dir.glob("texgyretermes-*.otf"):
            font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "TeX Gyre Termes",
                "Liberation Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
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


def clean_axes(ax, grid_axis: str = "y") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#9CA3AF")
        spine.set_linewidth(0.6)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def direction_control_panel(ax) -> None:
    controls = load_json("result/cprc_robustness/prior_controls_qwen32b_v1.json")[
        "experiments"
    ]["Qwen3-VL-32B-Instruct"]["prior_controls"]["summary"]

    specs = [
        ("full", "Aligned\n(primary)"),
        ("development_mean_prior", "Mean"),
        ("development_shuffled_prior", "Cross-\nquestion"),
        ("within_row_permuted_prior", "Permuted"),
        ("image_residual", "Image-\nonly"),
    ]
    x = np.arange(len(specs))
    width = 0.36

    cf = [100 * controls[key]["mc"]["counterfactual"]["delta"]["mean"] for key, _ in specs]
    cs = [100 * controls[key]["mc"]["commonsense"]["delta"]["mean"] for key, _ in specs]
    cf_err = [100 * controls[key]["mc"]["counterfactual"]["delta"]["std"] for key, _ in specs]
    cs_err = [100 * controls[key]["mc"]["commonsense"]["delta"]["std"] for key, _ in specs]

    err_kw = dict(elinewidth=0.8, capsize=2.2, capthick=0.8, ecolor=INK, zorder=4)
    bars_cf = ax.bar(
        x - width / 2,
        cf,
        width,
        color=GREEN,
        edgecolor="white",
        linewidth=0.6,
        label=r"$\Delta$CF (repair)",
        zorder=3,
    )
    bars_cs = ax.bar(
        x + width / 2,
        cs,
        width,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        label=r"$\Delta$CS (retention)",
        zorder=3,
    )
    for positions, values, errs in (
        (x - width / 2, cf, cf_err),
        (x + width / 2, cs, cs_err),
    ):
        noisy = [(pos, val, err) for pos, val, err in zip(positions, values, errs) if err > 0]
        if noisy:
            ax.errorbar(
                [item[0] for item in noisy],
                [item[1] for item in noisy],
                yerr=[item[2] for item in noisy],
                fmt="none",
                **err_kw,
            )

    ax.axhline(0, color=INK, lw=0.8, zorder=2)

    for bars, values, errs in ((bars_cf, cf, cf_err), (bars_cs, cs, cs_err)):
        is_cf = bars is bars_cf
        for bar, value, err in zip(bars, values, errs):
            if value == 0:
                # side-by-side zero bars: split labels above/below the axis
                ypos = 0.25 if is_cf else -0.25
                va = "bottom" if is_cf else "top"
            elif value > 0:
                ypos = value + err + 0.25
                va = "bottom"
            else:
                ypos = value - err - 0.25
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{value:+.1f}",
                ha="center",
                va=va,
                fontsize=8,
                color=INK,
            )

    ax.set_xticks(x, [label for _, label in specs])
    ax.set_xlim(-0.6, len(specs) - 0.4)
    ax.set_ylim(-3.6, 11.4)
    ax.set_yticks([-2, 0, 2, 4, 6, 8, 10])
    ax.set_ylabel(r"$\Delta$accuracy (points)", labelpad=2)
    ax.tick_params(axis="x", length=0, pad=2)
    clean_axes(ax, "y")

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        ncol=2,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.35,
        facecolor="white",
        edgecolor="#9CA3AF",
        framealpha=0.95,
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.6)

    ax.set_title(
        "(a) Prior-direction controls (Qwen3-VL-32B, MC)",
        loc="left",
        weight="bold",
        pad=4,
    )


def frontier_panel(ax) -> None:
    pareto = load_json("result/cprc_robustness/gate_pareto_qwen_llava_v1.json")[
        "experiments"
    ]
    budgets = ("0.00", "0.02", "0.04", "0.06", "0.10")
    models = [
        ("Qwen3-VL-32B-Instruct", "Qwen3-VL-32B", BLUE, "o", "-"),
        ("LLaVA-1.6-34B", "LLaVA-1.6-34B", ORANGE, "s", (0, (4, 2))),
    ]
    # budget tags: (x-offset, y-offset) in points, hand-tuned to avoid overlap
    budget_offsets = {
        "Qwen3-VL-32B-Instruct": {
            "0.00": (9, -2),
            "0.04": (-9, 6),
            "0.06": (-10, -7),
        },
        "LLaVA-1.6-34B": {
            "0.00": (9, -8),
            "0.02": (8, -4),
            "0.04": (-11, -9),
            "0.06": (-13, 1),
            "0.10": (0, 10),
        },
    }
    for model, label, color, marker, linestyle in models:
        selected = pareto[model]["selected_by_development_cs_budget"]
        points = []
        seen = set()
        for budget in budgets:
            result = selected["bprc_full"].get(budget)
            if result is None:
                continue
            metrics = result["test_metrics"]["test_mc"]
            point = (
                100 * metrics["commonsense"]["delta"],
                100 * metrics["counterfactual"]["delta"],
                budget,
            )
            points.append(point)
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=color,
            linewidth=1.6,
            linestyle=linestyle,
            marker=marker,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        for px, py, budget in points:
            if (round(px, 2), round(py, 2)) in seen:
                continue
            seen.add((round(px, 2), round(py, 2)))
            offset = budget_offsets.get(model, {}).get(budget)
            if offset is None:
                continue
            ax.annotate(
                budget,
                (px, py),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.5,
                color=GRAY,
                ha="center",
                va="center",
                zorder=4,
            )
        star = next(p for p in points if p[2] == "0.04")
        ax.scatter(
            star[0],
            star[1],
            marker="*",
            s=190,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        fixed = selected["fixed_lambda_same_gate"]["0.04"]["test_metrics"]["test_mc"]
        ax.scatter(
            100 * fixed["commonsense"]["delta"],
            100 * fixed["counterfactual"]["delta"],
            marker="D",
            s=42,
            facecolor="white",
            edgecolor=color,
            linewidth=1.3,
            zorder=4,
        )

    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    # extra left margin opens an empty lower-left zone for the legend, so it
    # no longer covers the frontier curves (data/values unchanged)
    ax.set_xlim(-3.9, 2.2)
    ax.set_ylim(-1.2, 15.2)
    ax.set_xticks([-3, -2, -1, 0, 1, 2])
    ax.set_xlabel(r"$\Delta$CS (points)", labelpad=2)
    ax.set_ylabel(r"$\Delta$CF (points)", labelpad=1)
    clean_axes(ax, "both")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            linewidth=1.6,
            linestyle="-",
            marker="o",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="Qwen3-VL-32B",
        ),
        Line2D(
            [0],
            [0],
            color=GRAY,
            linewidth=0,
            marker="*",
            markersize=11,
            markerfacecolor=GRAY,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=r"Primary (budget $0.04$)",
        ),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            linewidth=1.6,
            linestyle=(0, (4, 2)),
            marker="s",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="LLaVA-1.6-34B",
        ),
        Line2D(
            [0],
            [0],
            color=GRAY,
            linewidth=0,
            marker="D",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=GRAY,
            markeredgewidth=1.3,
            label=r"Constant $\lambda$ (same gate)",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        ncol=1,
        fontsize=8,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=0.9,
        borderpad=0.4,
        facecolor="white",
        edgecolor="#9CA3AF",
        framealpha=0.95,
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.6)

    ax.annotate(
        "point labels: CS budget",
        xy=(0.985, 0.975),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.5,
        color=GRAY,
        style="italic",
    )

    ax.set_title(
        "(b) CF-repair / CS-retention frontier (MC)",
        loc="left",
        weight="bold",
        pad=4,
    )


def main() -> None:
    setup()
    fig, (ax_control, ax_pareto) = plt.subplots(
        1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1.12, 1.0]}
    )
    direction_control_panel(ax_control)
    frontier_panel(ax_pareto)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.90, wspace=0.30)
    fig.savefig(FIGURES / "evidence_chain_v2.pdf")
    fig.savefig(FIGURES / "evidence_chain_v2.png", dpi=300)
    plt.close(fig)
    print(f"saved {FIGURES / 'evidence_chain_v2.pdf'}")


if __name__ == "__main__":
    main()
