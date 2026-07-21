import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[2]
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6B7280"
INK = "#1F2937"
GRID = "#E5E7EB"
LIGHT_BLUE = "#E8F3F8"
LIGHT_ORANGE = "#FBEDE7"
LIGHT_GREEN = "#E7F4EF"
LIGHT_GRAY = "#F3F4F6"
PURPLE = "#8E6BBE"


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
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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


def clean_axes(ax, grid_axis: str = "both") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#9CA3AF")
        spine.set_linewidth(0.6)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def box(ax, xy, width, height, text, face, edge, fontsize=9.5, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, color=GRAY, width=1.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=width, shrinkA=2, shrinkB=2),
    )


def save_panel(fig, stem: str) -> None:
    fig.tight_layout(pad=0.08)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(
        FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.015
    )
    plt.close(fig)


def mechanistic_questions_figure() -> None:
    fig, ax = plt.subplots(figsize=(1.58, 0.88))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    category_specs = [
        ((0.04, 0.66), "Count", LIGHT_BLUE, BLUE),
        ((0.53, 0.66), "Color / material", LIGHT_ORANGE, ORANGE),
        ((0.04, 0.40), "Relation", LIGHT_GREEN, GREEN),
        ((0.53, 0.40), "Logic", LIGHT_GRAY, GRAY),
    ]
    for xy, label, face, edge in category_specs:
        box(ax, xy, 0.43, 0.18, label, face, edge, fontsize=5.1, weight="bold")
    ax.text(
        0.43,
        0.14,
        "visual evidence",
        color=BLUE,
        ha="right",
        va="center",
        fontsize=5.0,
        weight="bold",
    )
    ax.text(0.50, 0.14, r"$\ne$", color=INK, ha="center", va="center", fontsize=8.5)
    ax.text(
        0.57,
        0.14,
        "world prior",
        color=ORANGE,
        ha="left",
        va="center",
        fontsize=5.0,
        weight="bold",
    )
    arrow(ax, (0.50, 0.38), (0.50, 0.22), GRAY, 0.9)
    save_panel(fig, "question_conflict_source")

    fig, ax = plt.subplots(figsize=(1.58, 0.88))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    candidates = np.arange(4)
    image_scores = np.asarray([0.12, 0.25, 0.51, 0.12])
    prior_scores = np.asarray([0.10, 0.62, 0.17, 0.11])
    x = 0.24 + candidates * 0.17
    for xpos, value in zip(x, image_scores):
        ax.plot([xpos, xpos], [0.61, 0.61 + value * 0.40], color=BLUE, lw=4.2)
    for xpos, value in zip(x, prior_scores):
        ax.plot([xpos, xpos], [0.23, 0.23 + value * 0.40], color=ORANGE, lw=4.2)
    ax.text(0.03, 0.72, "Image", color=BLUE, fontsize=5.4, weight="bold")
    ax.text(0.03, 0.34, "No image", color=ORANGE, fontsize=5.4, weight="bold")
    for xpos, label in zip(x, "ABCD"):
        ax.text(xpos, 0.11, label, ha="center", va="center", fontsize=5.1, color=GRAY)
    ax.annotate(
        "error",
        xy=(x[1], 0.23 + prior_scores[1] * 0.40),
        xytext=(0.82, 0.59),
        ha="center",
        va="center",
        fontsize=5.0,
        weight="bold",
        color=ORANGE,
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9),
    )
    ax.text(
        0.50,
        0.97,
        "70--92% of errors",
        ha="center",
        va="top",
        fontsize=5.0,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.50,
        0.87,
        r"$\rightarrow$ no-image top",
        ha="center",
        va="top",
        fontsize=4.9,
        weight="bold",
        color=INK,
    )
    save_panel(fig, "question_directed_attraction")

    fig, ax = plt.subplots(figsize=(1.58, 0.88))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(
        ax,
        (0.04, 0.43),
        0.41,
        0.42,
        "CF\nvision vs. prior\nprior hurts",
        LIGHT_ORANGE,
        ORANGE,
        fontsize=5.6,
        weight="bold",
    )
    box(
        ax,
        (0.55, 0.43),
        0.41,
        0.42,
        "CS\nvision + prior\nprior helps",
        LIGHT_GREEN,
        GREEN,
        fontsize=5.6,
        weight="bold",
    )
    arrow(ax, (0.25, 0.43), (0.43, 0.24), ORANGE, 0.9)
    arrow(ax, (0.75, 0.43), (0.57, 0.24), GREEN, 0.9)
    box(
        ax,
        (0.27, 0.07),
        0.46,
        0.17,
        "repair  /  retention",
        LIGHT_GRAY,
        GRAY,
        fontsize=5.5,
        weight="bold",
    )
    save_panel(fig, "question_dual_role")

    fig, ax = plt.subplots(figsize=(1.58, 0.88))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tests = [
        ((0.02, 0.69), "Candidate\nswap", LIGHT_BLUE, BLUE),
        ((0.66, 0.69), "Semantic\nswap", LIGHT_ORANGE, ORANGE),
        ((0.02, 0.08), "Family\nholdout", LIGHT_GREEN, GREEN),
        ((0.66, 0.08), "External\ntransfer", LIGHT_GRAY, GRAY),
    ]
    for xy, label, face, edge in tests:
        box(ax, xy, 0.32, 0.23, label, face, edge, fontsize=5.2, weight="bold")
    box(
        ax,
        (0.36, 0.36),
        0.28,
        0.25,
        "Reliable\nprior?",
        "white",
        INK,
        fontsize=5.8,
        weight="bold",
    )
    arrow(ax, (0.32, 0.73), (0.39, 0.59), BLUE, 0.8)
    arrow(ax, (0.68, 0.73), (0.61, 0.59), ORANGE, 0.8)
    arrow(ax, (0.32, 0.28), (0.39, 0.38), GREEN, 0.8)
    arrow(ax, (0.68, 0.28), (0.61, 0.38), GRAY, 0.8)
    save_panel(fig, "question_generality_boundary")


def method_figure() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 1.92))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        "Single-image inference",
        ha="left",
        va="top",
        fontsize=5.9,
        weight="bold",
        color=INK,
    )
    box(
        ax,
        (0.02, 0.66),
        0.20,
        0.17,
        "Image scores\n$I,Q,\\mathcal{Y}$",
        LIGHT_BLUE,
        BLUE,
        fontsize=5.8,
        weight="bold",
    )
    box(
        ax,
        (0.02, 0.41),
        0.20,
        0.17,
        "No-image prior\n$\\varnothing,Q,\\mathcal{Y}$",
        LIGHT_ORANGE,
        ORANGE,
        fontsize=5.7,
        weight="bold",
    )
    box(
        ax,
        (0.28, 0.48),
        0.25,
        0.34,
        "Candidate evidence\n$o=(\\bar s_I,\\bar s_0,i_b,K)$\nconflict, confidence",
        LIGHT_GRAY,
        GRAY,
        fontsize=5.5,
    )
    arrow(ax, (0.22, 0.75), (0.28, 0.70), BLUE)
    arrow(ax, (0.22, 0.50), (0.28, 0.59), ORANGE)

    box(
        ax,
        (0.58, 0.58),
        0.16,
        0.18,
        "Feature\nextractor\n$u=\\phi(o)$",
        LIGHT_BLUE,
        BLUE,
        fontsize=5.5,
        weight="bold",
    )
    box(
        ax,
        (0.79, 0.58),
        0.19,
        0.18,
        "Learned prior\ncalibrator\n$c_\\theta(u)\\rightarrow\\lambda$",
        LIGHT_GREEN,
        GREEN,
        fontsize=5.4,
        weight="bold",
    )
    arrow(ax, (0.52, 0.66), (0.58, 0.67), BLUE)
    arrow(ax, (0.74, 0.67), (0.79, 0.67), GREEN)

    ax.text(
        0.79,
        0.95,
        "Paired CF/CS fits $\\theta$ offline",
        ha="center",
        va="top",
        fontsize=5.5,
        weight="bold",
        color=GREEN,
    )
    ax.annotate(
        "",
        xy=(0.89, 0.77),
        xytext=(0.89, 0.88),
        arrowprops=dict(
            arrowstyle="-|>", color=GREEN, lw=0.9, linestyle="--", shrinkA=1, shrinkB=1
        ),
    )

    box(
        ax,
        (0.28, 0.29),
        0.70,
        0.14,
        "Candidate-aligned correction\n$s_C=\\bar s_I-c_\\theta(\\phi(o))\\bar s_0$",
        LIGHT_GREEN,
        GREEN,
        fontsize=5.5,
        weight="bold",
    )
    arrow(ax, (0.40, 0.48), (0.40, 0.43), GRAY)
    arrow(ax, (0.89, 0.58), (0.86, 0.43), GREEN)

    box(
        ax,
        (0.28, 0.06),
        0.22,
        0.14,
        "Proposal\n$z=\\arg\\max s_C$",
        LIGHT_BLUE,
        BLUE,
        fontsize=5.6,
        weight="bold",
    )
    box(
        ax,
        (0.57, 0.06),
        0.41,
        0.14,
        "Selective support gate\nreturn $z$ / keep native $i_b$",
        LIGHT_GRAY,
        GRAY,
        fontsize=5.5,
        weight="bold",
    )
    arrow(ax, (0.45, 0.29), (0.40, 0.20), BLUE)
    arrow(ax, (0.50, 0.13), (0.57, 0.13), GRAY)

    fig.tight_layout(pad=0.1)
    fig.savefig(FIGURES / "method_overview.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "method_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def evidence_figure() -> None:
    attribution = load_json("result/cprc_attribution/candidate_attribution_qwen_llava_v1.json")
    controls = load_json(
        "result/cprc_robustness/prior_controls_qwen32b_v1.json"
    )["experiments"]["Qwen3-VL-32B-Instruct"]["prior_controls"]
    pareto = load_json(
        "result/cprc_robustness/gate_pareto_qwen_llava_v1.json"
    )["experiments"]
    heldout_sources = [
        (
            "Qwen3-VL-32B-Instruct",
            "result/hierarchical_eb_lambda_32b/cprc_instruct_leave_one_supercategory_out_v1.json",
        ),
        (
            "LLaVA-1.6-34B",
            "result/hierarchical_eb_lambda_32b/cprc_llava16_34b_leave_one_supercategory_out_v1.json",
        ),
    ]
    visual_counterfact = load_json(
        "result/cprc_external/visual_counterfact_qwen_llava_v1.json"
    )
    hallusion = load_json(
        "result/cprc_external/hallusionbench_full_cdh_frozen_bf16_native_v1.json"
    )
    conflictvis = load_json(
        "result/cprc_external/conflictvis_full_frozen_bf16_v1.json"
    )
    pope = load_json("result/cprc_external/pope_native_full_cdh_frozen_bf16_v1.json")

    labels = ["Qwen MC", "Qwen QA", "LLaVA MC", "LLaVA QA"]
    model_task = [
        ("qwen3vl32b", "mc"),
        ("qwen3vl32b", "qa"),
        ("llava16_34b", "mc"),
        ("llava16_34b", "qa"),
    ]
    prior_match = [
        100
        * attribution["models"][model]["prior_alignment"][task][
            "native_counterfactual_errors"
        ]["baseline_equals_prior_top"]
        for model, task in model_task
    ]

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(7.05, 1.68),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.5, 1.0]},
    )
    ax_prior, ax_control, ax_pareto, ax_family, ax_external = axes
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.305,
        top=0.93,
        wspace=0.55,
    )

    # A: a dot plot emphasizes convergence on a concrete candidate without
    # implying that a truncated bar starts at zero.
    y = np.arange(len(labels))[::-1]
    point_colors = [BLUE, BLUE, ORANGE, ORANGE]
    point_markers = ["o", "^", "o", "^"]
    for row, value, color, marker in zip(y, prior_match, point_colors, point_markers):
        ax_prior.hlines(row, 60, value, color=GRID, linewidth=2.0, zorder=1)
        ax_prior.scatter(
            value,
            row,
            s=38,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax_prior.text(value + 1.0, row, f"{value:.1f}", va="center", fontsize=7.4)
    ax_prior.set_xlim(60, 100)
    ax_prior.set_xticks([60, 80, 100])
    ax_prior.set_yticks(y, labels)
    ax_prior.set_xlabel("Prior-match errors (%)", fontsize=6.2, labelpad=2)
    ax_prior.tick_params(axis="both", length=0, labelsize=6.0)
    clean_axes(ax_prior, "x")

    control_names = [
        ("full", "Aligned", GREEN),
        ("development_mean_prior", "Mean", BLUE),
        ("development_shuffled_prior", "Cross-Q", ORANGE),
        ("within_row_permuted_prior", "Permuted", PURPLE),
    ]
    label_offsets = {
        "Aligned": (3, 3),
        "Mean": (3, -2),
        "Cross-Q": (3, -2),
        "Permuted": (-3, 4),
    }
    for name, label, color in control_names:
        summary = controls["summary"][name]
        task_points = []
        for task, marker in (("mc", "o"), ("qa", "^")):
            point = (
                100 * summary[task]["commonsense"]["delta"]["mean"],
                100 * summary[task]["counterfactual"]["delta"]["mean"],
            )
            task_points.append(point)
            ax_control.scatter(
                point[0],
                point[1],
                color=color,
                marker=marker,
                s=30,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )
        ax_control.plot(
            [point[0] for point in task_points],
            [point[1] for point in task_points],
            color=color,
            linewidth=0.8,
            alpha=0.65,
            zorder=2,
        )
        anchor = task_points[1] if label != "Permuted" else task_points[0]
        ax_control.annotate(
            label,
            anchor,
            xytext=label_offsets[label],
            textcoords="offset points",
            fontsize=5.7,
            color=color,
            weight="bold" if label == "Aligned" else "normal",
            ha="right" if label == "Permuted" else "left",
        )
    ax_control.axhline(0, color=INK, lw=0.7)
    ax_control.axvline(0, color=INK, lw=0.7)
    ax_control.set_xlim(-7.0, 1.6)
    ax_control.set_ylim(-3.0, 27.0)
    ax_control.set_xlabel(r"$\Delta$CS (points)", fontsize=6.2, labelpad=2)
    ax_control.set_ylabel(r"$\Delta$CF (points)", fontsize=6.2, labelpad=1)
    ax_control.tick_params(axis="both", labelsize=5.8)
    clean_axes(ax_control)

    # C: development-selected policies trace the paired repair-retention
    # frontier. The open diamond is the matched fixed-coefficient policy.
    budgets = ("0.00", "0.02", "0.04", "0.06", "0.10")
    pareto_models = [
        ("Qwen3-VL-32B-Instruct", "Qwen", BLUE),
        ("LLaVA-1.6-34B", "LLaVA", ORANGE),
    ]
    for model, label, color in pareto_models:
        selected = pareto[model]["selected_by_development_cs_budget"]
        points = []
        for budget in budgets:
            result = selected["bprc_full"].get(budget)
            if result is None:
                continue
            metrics = result["test_metrics"]["test_mc"]
            points.append(
                (
                    100 * metrics["commonsense"]["delta"],
                    100 * metrics["counterfactual"]["delta"],
                    budget,
                )
            )
        ax_pareto.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        selected_point = next(point for point in points if point[2] == "0.04")
        ax_pareto.scatter(
            selected_point[0],
            selected_point[1],
            marker="*",
            s=90,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        fixed = selected["fixed_lambda_same_gate"]["0.04"]["test_metrics"]["test_mc"]
        ax_pareto.scatter(
            100 * fixed["commonsense"]["delta"],
            100 * fixed["counterfactual"]["delta"],
            marker="D",
            s=34,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=4,
        )
        endpoint = points[-1]
        ax_pareto.annotate(
            label,
            (endpoint[0], endpoint[1]),
            xytext=(5, 3 if label == "Qwen" else -4),
            textcoords="offset points",
            va="center",
            fontsize=5.8,
            color=color,
            weight="bold",
        )
    ax_pareto.axhline(0, color=INK, lw=0.7)
    ax_pareto.axvline(0, color=INK, lw=0.7)
    ax_pareto.set_xlim(-3.0, 1.5)
    ax_pareto.set_ylim(-0.5, 14.2)
    ax_pareto.set_xlabel(r"$\Delta$CS (points)", fontsize=6.2, labelpad=2)
    ax_pareto.set_ylabel(r"$\Delta$CF (points)", fontsize=6.2, labelpad=1)
    ax_pareto.tick_params(axis="both", labelsize=5.8)
    clean_axes(ax_pareto)

    families = ["Attribute Anomalies", "Counting Anomalies", "Relational Anomalies"]
    row_specs = [
        ("Qwen3-VL-32B-Instruct", "mc", "Qwen MC"),
        ("Qwen3-VL-32B-Instruct", "qa", "Qwen QA"),
        ("LLaVA-1.6-34B", "mc", "LLaVA MC"),
        ("LLaVA-1.6-34B", "qa", "LLaVA QA"),
    ]
    fold_lookup = {}
    for model, path in heldout_sources:
        folds = load_json(path)["result"]["folds"]
        fold_lookup[model] = {fold["heldout_group"]: fold for fold in folds}
    family_values = {}
    for row_index, (model, task, _) in enumerate(row_specs):
        for col_index, family in enumerate(families):
            metrics = fold_lookup[model][family]["metrics"][task]
            delta_cf = 100 * metrics["counterfactual"]["delta"]
            delta_cs = 100 * metrics["commonsense"]["delta"]
            family_values[(row_index, col_index)] = (delta_cf, delta_cs)

    max_family_cf = max(value[0] for value in family_values.values())
    ax_family.set_xlim(-0.5, 2.5)
    ax_family.set_ylim(3.5, -0.5)
    ax_family.set_facecolor("#F8FAFC")
    for row_index in range(len(row_specs)):
        for col_index in range(len(families)):
            delta_cf, delta_cs = family_values[(row_index, col_index)]
            tile = FancyBboxPatch(
                (col_index - 0.42, row_index - 0.38),
                0.84,
                0.76,
                boxstyle="round,pad=0.018,rounding_size=0.08",
                linewidth=0.55,
                facecolor="white",
                edgecolor="#D1D5DB",
                zorder=1,
            )
            ax_family.add_patch(tile)
            ax_family.text(
                col_index - 0.03,
                row_index - 0.04,
                f"{delta_cf:+.1f}",
                ha="right",
                va="center",
                fontsize=4.8,
                color=GREEN,
                weight="bold",
            )
            ax_family.text(
                col_index,
                row_index - 0.04,
                "/",
                ha="center",
                va="center",
                fontsize=4.7,
                color=GRAY,
                zorder=3,
            )
            ax_family.text(
                col_index + 0.03,
                row_index - 0.04,
                f"{delta_cs:+.1f}",
                ha="left",
                va="center",
                fontsize=4.8,
                color=ORANGE if delta_cs < -0.05 else GRAY,
                zorder=3,
            )
            track_x = col_index - 0.32
            track_y = row_index + 0.24
            track_width = 0.64
            ax_family.plot(
                [track_x, track_x + track_width],
                [track_y, track_y],
                color="#E5E7EB",
                linewidth=1.8,
                solid_capstyle="round",
                zorder=2,
            )
            ax_family.plot(
                [track_x, track_x + track_width * delta_cf / max_family_cf],
                [track_y, track_y],
                color=GREEN,
                linewidth=1.8,
                solid_capstyle="round",
                zorder=3,
            )
    ax_family.set_xticks(
        range(3),
        ["Attribute\nAnomalies", "Counting\nAnomalies", "Relational\nAnomalies"],
    )
    ax_family.set_yticks(range(4), ["Q-MC", "Q-QA", "L-MC", "L-QA"])
    ax_family.tick_params(axis="both", length=0, labelsize=6.0, pad=1)
    for spine in ax_family.spines.values():
        spine.set_visible(True)
        spine.set_color("#9CA3AF")
        spine.set_linewidth(0.6)

    def visual_counterfact_pair(model: str) -> tuple[float, float]:
        selected = visual_counterfact["experiments"][model]["selected"][
            "map_no_laplace_or_density"
        ]["0.04"]
        metrics = selected["by_stratum"]["overall"]["metrics"]
        return (
            100 * metrics["counterfactual"]["delta"],
            100 * metrics["commonsense"]["delta"],
        )

    hallusion_metrics = hallusion["result"]["evaluation_by_dataset"]["hallusion_full"]
    external_rows = [
        ("VCF-Q", *visual_counterfact_pair("Qwen3-VL-32B-Instruct"), True),
        ("VCF-L", *visual_counterfact_pair("LLaVA-1.6-34B"), True),
        (
            "Hallusion",
            100 * hallusion_metrics["counterfactual"]["delta"],
            100 * hallusion_metrics["commonsense"]["delta"],
            True,
        ),
        (
            "ConflictVIS",
            100 * conflictvis["metrics"]["task:mc"]["eb_cprc"]["delta_accuracy"],
            np.nan,
            False,
        ),
        (
            "POPE",
            100 * pope["results"]["overall"]["delta_accuracy"],
            np.nan,
            False,
        ),
    ]
    row_y = np.arange(len(external_rows))[::-1]
    for ypos, (label, conflict_delta, control_delta, paired) in zip(row_y, external_rows):
        if paired:
            ax_external.plot(
                [control_delta, conflict_delta],
                [ypos, ypos],
                color="#CBD5E1",
                linewidth=2.0,
                zorder=1,
            )
            ax_external.scatter(
                control_delta,
                ypos,
                marker="s",
                s=27,
                color=ORANGE,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        ax_external.scatter(
            conflict_delta,
            ypos,
            marker="o" if label != "POPE" else "X",
            s=34,
            color=GREEN if label != "POPE" else GRAY,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        if label == "POPE":
            ax_external.annotate(
                "abstain",
                (conflict_delta, ypos),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=5.8,
                color=GRAY,
            )
    ax_external.axvline(0, color=INK, lw=0.7)
    ax_external.set_xlim(-3.2, 6.0)
    ax_external.set_ylim(-0.85, 4.25)
    ax_external.set_yticks(row_y, [row[0] for row in external_rows])
    ax_external.tick_params(axis="both", length=0, labelsize=6.0, pad=1)
    ax_external.set_xlabel(r"$\Delta$accuracy (points)", fontsize=6.5, labelpad=2)
    clean_axes(ax_external, "x")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_outputs = [
        (ax_prior, "evidence_prior_attraction"),
        (ax_control, "evidence_candidate_alignment"),
        (ax_pareto, "evidence_repair_retention"),
        (ax_family, "evidence_semantic_holdout"),
        (ax_external, "evidence_frozen_transfer"),
    ]
    regular_axes = [ax_prior, ax_control, ax_pareto, ax_external]
    axis_extents = {
        panel_ax: panel_ax.get_window_extent(renderer).transformed(
            fig.dpi_scale_trans.inverted()
        )
        for panel_ax in regular_axes
    }
    tight_extents = {
        panel_ax: panel_ax.get_tightbbox(renderer).transformed(
            fig.dpi_scale_trans.inverted()
        )
        for panel_ax in regular_axes
    }
    left_margin = max(
        axis_extents[panel_ax].x0 - tight_extents[panel_ax].x0
        for panel_ax in regular_axes
    )
    right_margin = max(
        tight_extents[panel_ax].x1 - axis_extents[panel_ax].x1
        for panel_ax in regular_axes
    )
    bottom_margin = max(
        axis_extents[panel_ax].y0 - tight_extents[panel_ax].y0
        for panel_ax in regular_axes
    )
    top_margin = max(
        tight_extents[panel_ax].y1 - axis_extents[panel_ax].y1
        for panel_ax in regular_axes
    )
    regular_extents = {}
    for panel_ax in regular_axes:
        axis_extent = axis_extents[panel_ax]
        regular_extents[panel_ax] = Bbox.from_extents(
            axis_extent.x0 - left_margin - 0.025,
            axis_extent.y0 - bottom_margin - 0.025,
            axis_extent.x1 + right_margin + 0.025,
            axis_extent.y1 + top_margin + 0.025,
        )
    panel_extents = dict(regular_extents)
    panel_extents[ax_family] = ax_family.get_tightbbox(renderer).transformed(
        fig.dpi_scale_trans.inverted()
    ).padded(0.025)
    for panel_ax, stem in panel_outputs:
        extent = panel_extents[panel_ax]
        for other_ax in axes:
            other_ax.set_visible(other_ax is panel_ax)
        fig.savefig(
            FIGURES / f"{stem}.pdf",
            bbox_inches=extent,
            pad_inches=0.0,
        )
        fig.savefig(
            FIGURES / f"{stem}.png",
            dpi=300,
            bbox_inches=extent,
            pad_inches=0.0,
        )
    for panel_ax in axes:
        panel_ax.set_visible(True)
    fig.savefig(FIGURES / "evidence_chain.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "evidence_chain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def calibration_size_figure() -> None:
    experiments = load_json(
        "result/cprc_robustness/calibration_size_qwen_llava_v1.json"
    )["experiments"]
    panels = [
        ("Qwen3-VL-32B-Instruct", "mc", "Qwen3-VL-32B, MC"),
        ("Qwen3-VL-32B-Instruct", "qa", "Qwen3-VL-32B, QA"),
        ("LLaVA-1.6-34B", "mc", "LLaVA-1.6-34B, MC"),
        ("LLaVA-1.6-34B", "qa", "LLaVA-1.6-34B, QA"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True)
    x = np.asarray([14, 28, 42, 56, 70])
    for ax, (model, task, title) in zip(axes.flat, panels):
        summaries = experiments[model]["summary_by_pairs_per_subcategory"]
        cf = [summaries[str(size // 14)][task]["counterfactual"]["delta"] for size in x]
        cs = [summaries[str(size // 14)][task]["commonsense"]["delta"] for size in x]
        ax.errorbar(
            x,
            [100 * value["mean"] for value in cf],
            yerr=[100 * value["std"] for value in cf],
            color=GREEN,
            marker="o",
            capsize=2.5,
            linewidth=1.4,
            label=r"$\Delta$CF",
        )
        ax.errorbar(
            x,
            [100 * value["mean"] for value in cs],
            yerr=[100 * value["std"] for value in cs],
            color=ORANGE,
            marker="s",
            capsize=2.5,
            linewidth=1.2,
            label=r"$\Delta$CS",
        )
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylabel("Accuracy change (points)")
        ax.set_xticks(x)
        ax.grid(axis="y", color="#D1D5DB", lw=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.supxlabel("Calibration pairs (stratified across 14 subcategories)", y=0.01)
    axes[0, 0].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=1.0, h_pad=1.0)
    fig.savefig(FIGURES / "calibration_size.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "calibration_size.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def repeated_splits_figure() -> None:
    experiments = load_json(
        "result/cprc_robustness/repeated_splits_qwen_llava_v1.json"
    )["experiments"]
    series = [
        ("Qwen3-VL-32B-Instruct", "mc", "Qwen MC", BLUE, "o"),
        ("Qwen3-VL-32B-Instruct", "qa", "Qwen QA", GREEN, "^"),
        ("LLaVA-1.6-34B", "mc", "LLaVA MC", ORANGE, "s"),
        ("LLaVA-1.6-34B", "qa", "LLaVA QA", PURPLE, "D"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    for model, task, label, color, marker in series:
        trials = experiments[model]["repeated_splits"]["trials"]
        delta_cs = [
            100 * trial["tasks"][task]["commonsense"]["delta"] for trial in trials
        ]
        delta_cf = [
            100 * trial["tasks"][task]["counterfactual"]["delta"] for trial in trials
        ]
        ax.scatter(
            delta_cs,
            delta_cf,
            color=color,
            marker=marker,
            s=34,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.85,
            label=label,
        )
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel(r"$\Delta$CS (points)")
    ax.set_ylabel(r"$\Delta$CF (points)")
    ax.grid(color="#D1D5DB", lw=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(
        frameon=True,
        ncol=2,
        fontsize=8,
        loc="upper left",
        facecolor="white",
        edgecolor="#9CA3AF",
        framealpha=0.96,
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.6)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGURES / "repeated_splits.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "repeated_splits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def gate_pareto_figure() -> None:
    experiments = load_json(
        "result/cprc_robustness/gate_pareto_qwen_llava_v1.json"
    )["experiments"]
    panels = [
        ("Qwen3-VL-32B-Instruct", "mc", "Qwen3-VL-32B\nMC"),
        ("Qwen3-VL-32B-Instruct", "qa", "Qwen3-VL-32B\nQA"),
        ("LLaVA-1.6-34B", "mc", "LLaVA-1.6-34B\nMC"),
        ("LLaVA-1.6-34B", "qa", "LLaVA-1.6-34B\nQA"),
    ]
    styles = [
        ("bprc_full", "CPRC", GREEN, "o"),
        ("fixed_lambda_same_gate", r"Fixed $\lambda$ + CPRC gate", BLUE, "s"),
        ("pai_adapted_ungated", "PAI-adapted", ORANGE, "D"),
        ("pai_adapted_13d_gate", "PAI-adapted + 13D gate", PURPLE, "^"),
        ("fixed_lambda_13d_gate", r"Fixed $\lambda$ + 13D gate", GRAY, "v"),
    ]
    budgets = ("0.00", "0.02", "0.04", "0.06", "0.10")
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
    for ax, (model, task, title) in zip(axes.flat, panels):
        selected = experiments[model]["selected_by_development_cs_budget"]
        for family, label, color, marker in styles:
            points = []
            for budget in budgets:
                result = selected.get(family, {}).get(budget)
                if result is None:
                    continue
                metrics = result["test_metrics"][f"test_{task}"]
                points.append(
                    (
                        100 * metrics["commonsense"]["delta"],
                        100 * metrics["counterfactual"]["delta"],
                        budget,
                    )
                )
            if not points:
                continue
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.1,
                label=label,
            )
            if family == "bprc_full":
                grouped = {}
                for delta_cs, delta_cf, budget in points:
                    grouped.setdefault((delta_cs, delta_cf), []).append(
                        str(int(round(100 * float(budget))))
                    )
                for (delta_cs, delta_cf), point_budgets in grouped.items():
                    ax.annotate(
                        ",".join(point_budgets),
                        (delta_cs, delta_cf),
                        xytext=(3, 3),
                        textcoords="offset points",
                        color=color,
                        fontsize=6.5,
                    )
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(title, loc="center", weight="bold", fontsize=8.5, pad=2)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(color="#D1D5DB", lw=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = []
    labels = []
    for ax in axes.flat:
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        handles.extend(panel_handles)
        labels.extend(panel_labels)
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=6.4,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.35,
    )
    fig.supxlabel(r"Blind-test $\Delta$CS (points)", y=0.01, fontsize=8)
    fig.supylabel(r"Blind-test $\Delta$CF (points)", x=0.006, fontsize=8)
    fig.tight_layout(rect=(0.025, 0.08, 1, 0.86), w_pad=0.55)
    fig.savefig(FIGURES / "gate_pareto.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "gate_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    setup()
    mechanistic_questions_figure()
    method_figure()
    evidence_figure()
    calibration_size_figure()
    repeated_splits_figure()
    gate_pareto_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
