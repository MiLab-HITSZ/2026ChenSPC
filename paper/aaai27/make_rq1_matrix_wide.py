"""Wide (transposed) variant of the RQ1 subcategory matrix.

Same data and aggregation as make_rq1_matrix.py (native CF errors whose wrong
answer equals prior_top), but laid out to sit next to evidence_chain_v2 in one
figure row: 14 subcategories as columns (grouped by the three semantic
families, same within-family order as the tall version) x 4 model/task cells
as rows. Target size ~2.7 x 1.7 in.

Cells show the hit rate only; small-sample cells (n < 5) are italic. No
colorbar (values are annotated; color is redundant) to save width.

Outputs: figures/rq1_subcategory_matrix_wide.pdf / .png
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory

from make_rq1_matrix import (
    ATTRIBUTION,
    BLUE,
    FAMILIES,
    GRAY,
    INK,
    ROOT,
    FIGURES,
    aggregate,
    setup,
)

SHORT_COL_LABELS = {
    "Temperature": "Temp.",
    "Material": "Material",
    "Color": "Color",
    "Luminescence/Transparency": "Luminesc.",
    "Physical State": "Phys. State",
    "Everyday Objects": "Everyday Obj.",
    "Body Parts": "Body Parts",
    "Animal Parts": "Animal Parts",
    "Plant Structure": "Plant Struct.",
    "Object Function": "Object Func.",
    "Spatial": "Spatial",
    "Animal Behavior": "Animal Behav.",
    "Causality": "Causality",
    "Size Scale": "Size Scale",
}

ROW_SPECS = [
    ("Qwen MC", "qwen3vl32b", "mc"),
    ("Qwen QA", "qwen3vl32b", "qa"),
    ("LLaVA MC", "llava16_34b", "mc"),
    ("LLaVA QA", "llava16_34b", "qa"),
]

SMALL_N = 5  # cells with fewer native CF errors are italicized


def main() -> None:
    setup()

    attribution = json.loads(ATTRIBUTION.read_text(encoding="utf-8"))
    data = {
        key: aggregate(ROOT / attribution["models"][key]["results"])
        for key in ("qwen3vl32b", "llava16_34b")
    }

    # validate the four totals against the attribution JSON
    for model_key, per in data.items():
        expected = attribution["models"][model_key]["prior_alignment"]
        for task in ("mc", "qa"):
            hits = sum(v[0] for (sub, t), v in per.items() if t == task)
            n = sum(v[1] for (sub, t), v in per.items() if t == task)
            ref = expected[task]["native_counterfactual_errors"]
            assert n == ref["n"] and abs(hits / n - ref["baseline_equals_prior_top"]) < 1e-9
    print("validated four totals against attribution JSON")

    # column order: same family grouping and within-family order as tall version
    def mean_of(sub):
        vals = [
            data[m][(sub, t)][0] / data[m][(sub, t)][1]
            for m in data
            for t in ("mc", "qa")
            if data[m][(sub, t)][1]
        ]
        return sum(vals) / len(vals) if vals else 0.0

    cols = []
    family_spans = []
    for family, subs in FAMILIES:
        ordered = sorted(subs, key=mean_of, reverse=True)
        start = len(cols)
        cols.extend(ordered)
        family_spans.append((family, start, len(cols) - 1))

    n_rows, n_cols = len(ROW_SPECS), len(cols)
    values = np.full((n_rows, n_cols), np.nan)
    counts = np.zeros((n_rows, n_cols), dtype=int)
    for i, (_, model_key, task) in enumerate(ROW_SPECS):
        for j, sub in enumerate(cols):
            h, n = data[model_key][(sub, task)]
            counts[i, j] = n
            if n:
                values[i, j] = h / n

    cmap = LinearSegmentedColormap.from_list("prior_attraction", ["#FFFFFF", BLUE])
    fig, ax = plt.subplots(figsize=(2.7, 1.7))
    ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            value, n = values[i, j], counts[i, j]
            if n == 0:
                ax.text(j, i, "--", ha="center", va="center", fontsize=5.5, color=GRAY)
                continue
            dark = value > 0.62
            small = n < SMALL_N
            color = "white" if dark else (GRAY if small else INK)
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                fontsize=5.6, color=color, style="italic" if small else "normal",
            )

    # grid lines; heavier separators between families
    boundaries = [0]
    for _, start, end in family_spans:
        boundaries.append(end + 1)
    for k in range(n_cols + 1):
        heavy = k in boundaries
        ax.axvline(k - 0.5, color=INK if heavy else "white", lw=0.9 if heavy else 0.5, zorder=3)
    for k in range(n_rows + 1):
        ax.axhline(k - 0.5, color="white", lw=0.6, zorder=3)

    # column headers: rotated subcategory names on top
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [SHORT_COL_LABELS.get(s, s) for s in cols],
        rotation=90, fontsize=5.4, ha="center", va="bottom",
    )
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([spec[0] for spec in ROW_SPECS], fontsize=6.2)

    # family names centered above the rotated column labels
    trans_fam = blended_transform_factory(ax.transData, ax.transAxes)
    for family, start, end in family_spans:
        center = (start + end) / 2
        ax.text(
            center, 1.58, family, transform=trans_fam, ha="center", va="bottom",
            fontsize=6.0, weight="bold",
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.205, right=0.995, top=0.60, bottom=0.035)
    fig.savefig(FIGURES / "rq1_subcategory_matrix_wide.pdf")
    fig.savefig(FIGURES / "rq1_subcategory_matrix_wide.png", dpi=300)
    plt.close(fig)
    print(f"saved {FIGURES / 'rq1_subcategory_matrix_wide.pdf'}")


if __name__ == "__main__":
    main()
