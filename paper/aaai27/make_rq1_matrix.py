"""RQ1 subcategory matrix: directed prior attraction by CDH-Bench subcategory.

For each model x task cell, the value is the fraction of native
counterfactual errors whose (wrong) answer equals the prior-estimate winner
(prior_top) -- i.e. the strength of directed prior attraction.

Data:
  Qwen3-VL-32B  result/cprc_paper_cpr_unseen_32b_instruct/.../results.option_repaired.jsonl
  LLaVA-1.6-34B result/cprc_llava16_34b_cpr_unseen/.../results.jsonl
  (LLaVA path taken from result/cprc_attribution/candidate_attribution_qwen_llava_v1.json)

Native CF error = side == "counterfactual" and baseline_key != gt.
Hit = baseline_key == prior_top. The four column totals are validated against
models.*.prior_alignment.{mc,qa}.native_counterfactual_errors.baseline_equals_prior_top
in the attribution JSON.

Outputs: figures/rq1_subcategory_matrix.pdf / .png
Style follows make_paper_figures.py (TeX Gyre Termes, 0.7 pt axes, fonttype 42).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[2]
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ATTRIBUTION = ROOT / "result/cprc_attribution/candidate_attribution_qwen_llava_v1.json"

BLUE = "#0072B2"
GRAY = "#6B7280"
INK = "#1F2937"
GRID = "#E5E7EB"

FAMILIES = [
    ("Attributes", ["Color", "Material", "Temperature", "Physical State", "Luminescence/Transparency"]),
    ("Counting", ["Body Parts", "Animal Parts", "Plant Structure", "Everyday Objects"]),
    ("Relations", ["Animal Behavior", "Object Function", "Spatial", "Size Scale", "Causality"]),
]

SHORT_LABELS = {
    "Luminescence/Transparency": "Luminesc./Transp.",
    "Everyday Objects": "Everyday Objects",
    "Plant Structure": "Plant Structure",
    "Animal Behavior": "Animal Behavior",
    "Object Function": "Object Function",
}


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


def aggregate(path: Path):
    """Return per[(subcategory, task)] = [hits, n] over native CF errors."""
    per = defaultdict(lambda: [0, 0])
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("side") != "counterfactual":
                continue
            cp = row.get("cp_vbc") or {}
            pred, prior, gt = cp.get("baseline_key"), cp.get("prior_top"), row.get("gt")
            if pred is None or prior is None or gt is None:
                continue
            if pred == gt:
                continue  # not a native error
            cell = per[(row["subcategory"], row["task"])]
            cell[1] += 1
            cell[0] += pred == prior
    return per


def main() -> None:
    setup()

    attribution = json.loads(ATTRIBUTION.read_text(encoding="utf-8"))
    qwen_path = ROOT / attribution["models"]["qwen3vl32b"]["results"]
    llava_path = ROOT / attribution["models"]["llava16_34b"]["results"]
    qwen = aggregate(qwen_path)
    llava = aggregate(llava_path)

    # ---- validation against the attribution JSON four totals ----
    for model_key, per in (("qwen3vl32b", qwen), ("llava16_34b", llava)):
        expected = attribution["models"][model_key]["prior_alignment"]
        for task in ("mc", "qa"):
            hits = sum(v[0] for (sub, t), v in per.items() if t == task)
            n = sum(v[1] for (sub, t), v in per.items() if t == task)
            ref = expected[task]["native_counterfactual_errors"]
            assert n == ref["n"], (model_key, task, n, ref["n"])
            assert abs(hits / n - ref["baseline_equals_prior_top"]) < 1e-9, (
                model_key,
                task,
                hits / n,
                ref["baseline_equals_prior_top"],
            )
            print(
                f"validated {model_key} {task}: {hits}/{n} = {hits / n:.4f} "
                f"(ref {ref['baseline_equals_prior_top']:.4f})"
            )

    # ---- row order: families, sorted within family by column mean ----
    def mean_of(sub):
        vals = []
        for per in (qwen, llava):
            for task in ("mc", "qa"):
                h, n = per[(sub, task)]
                if n:
                    vals.append(h / n)
        return sum(vals) / len(vals) if vals else 0.0

    rows = []
    family_spans = []
    for family, subs in FAMILIES:
        ordered = sorted(subs, key=mean_of, reverse=True)
        start = len(rows)
        rows.extend(ordered)
        family_spans.append((family, start, len(rows) - 1))

    columns = [
        ("Qwen3-VL-32B", "MC", qwen, "mc"),
        ("Qwen3-VL-32B", "QA", qwen, "qa"),
        ("LLaVA-1.6-34B", "MC", llava, "mc"),
        ("LLaVA-1.6-34B", "QA", llava, "qa"),
    ]

    values = np.zeros((len(rows), len(columns)))
    counts = np.zeros_like(values, dtype=int)
    for i, sub in enumerate(rows):
        for j, (_, _, per, task) in enumerate(columns):
            h, n = per[(sub, task)]
            values[i, j] = h / n if n else np.nan
            counts[i, j] = n

    # ---- figure ----
    cmap = LinearSegmentedColormap.from_list("prior_attraction", ["#FFFFFF", BLUE])
    fig, ax = plt.subplots(figsize=(3.3, 3.6))
    ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(len(rows)):
        for j in range(len(columns)):
            value, n = values[i, j], counts[i, j]
            if n == 0:
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color=GRAY)
                continue
            dark = value > 0.62
            value_color = "white" if dark else INK
            n_color = "#E3EDF4" if dark else GRAY
            ax.text(
                j, i - 0.16, f"{value:.2f}", ha="center", va="center",
                fontsize=8, color=value_color,
            )
            ax.text(
                j, i + 0.24, f"n={n}", ha="center", va="center",
                fontsize=5.8, color=n_color,
            )

    # cell grid + family separators
    for k in range(len(rows) + 1):
        lw = 1.1 if k in (0, 5, 9, 14) else 0.5
        color = INK if lw > 1 else "white"
        ax.axhline(k - 0.5, color=color, lw=lw, zorder=3)
    for k in range(len(columns) + 1):
        ax.axvline(k - 0.5, color="white", lw=0.8, zorder=3)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[1] for c in columns], fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT_LABELS.get(s, s) for s in rows], fontsize=7.6)

    # model names spanning column pairs, above the MC/QA ticks
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x0, name in ((0, "Qwen3-VL-32B"), (2, "LLaVA-1.6-34B")):
        ax.text(
            x0 + 0.5, 1.115, name, transform=trans, ha="center", va="bottom",
            fontsize=7.2, weight="bold",
        )

    # family labels rotated on the left
    trans_fam = blended_transform_factory(ax.transAxes, ax.transData)
    for family, start, end in family_spans:
        ax.text(
            -0.60, (start + end) / 2, family, transform=trans_fam,
            ha="center", va="center", rotation=90, fontsize=8.2, weight="bold",
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        ax.images[0], ax=ax, fraction=0.032, pad=0.03, ticks=[0, 0.25, 0.5, 0.75, 1.0]
    )
    cbar.ax.tick_params(labelsize=7, length=2, width=0.6)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("prior-attraction rate", fontsize=7.5, labelpad=2)

    fig.subplots_adjust(left=0.335, right=0.82, top=0.855, bottom=0.02)
    fig.savefig(FIGURES / "rq1_subcategory_matrix.pdf")
    fig.savefig(FIGURES / "rq1_subcategory_matrix.png", dpi=300)
    plt.close(fig)
    print(f"saved {FIGURES / 'rq1_subcategory_matrix.pdf'}")


if __name__ == "__main__":
    main()
