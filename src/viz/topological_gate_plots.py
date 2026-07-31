"""Figures for the TOM (Topologically-Ordered Memory) gate result.

Panel A: the SOM collapse across the sigma sweep / loss ablation -- every
configuration with a meaningful SOM term drops to ARI 0.0000.

Panel B: the finding that actually decides the plan. Per-subject, the EXISTING
(non-topological) model's implicit ordinal axis (|Spearman| of embedding PC1
vs. true cortical layer order) plotted against that subject's ARI. Subject 3 --
the persistent weak point TOM was designed to fix -- has by far the STRONGEST
and most stable ordinal axis while having the WORST ARI. Strong laminar
ordering, weak clustering: subject 3's failure is not a missing depth order,
so an explicit ordinal prior targets a deficit it does not have.

Reads only already-logged JSON; no training here.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"

SUBJECT_COLORS = {"subject1": "#2c5f8a", "subject2": "#4f8fc0", "subject3": "#c0392b"}


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sweep = json.loads((LOGS_DIR / "topological_sweep.json").read_text())
    axis = json.loads((LOGS_DIR / "baseline_ordinal_axis.json").read_text())
    multislice = json.loads((LOGS_DIR / "dlpfc_multislice_results.json").read_text())

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel A: SOM collapse -------------------------------------------------
    labels = [r["label"] for r in sweep]
    aris = [r["ari"] for r in sweep]
    collapsed = [r["n_slots_used"] == 1 for r in sweep]
    colors = ["#c0392b" if c else "#2c5f8a" for c in collapsed]

    y = np.arange(len(labels))
    ax_a.barh(y, aris, color=colors, edgecolor="black", linewidth=0.5)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=9)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("ARI on DLPFC 151673 (seed 0)")
    ax_a.set_title("A. Every SOM-enabled config collapses to ARI 0.000")
    # The collapsed bars are zero-width, so the failure needs labelling
    # explicitly or the panel just looks like missing data.
    for yi, (a, c) in enumerate(zip(aris, collapsed)):
        if c:
            ax_a.text(0.008, yi, "collapsed to 1 slot (ARI 0.000)", va="center",
                      fontsize=8, color="#c0392b", fontweight="bold")
    baseline_ari = next(r["ari"] for r in sweep if r["label"].startswith("neither"))
    ax_a.axvline(baseline_ari, color="black", linestyle="--", linewidth=1.2,
                 label=f"Stage 13 baseline = {baseline_ari:.3f}")
    ax_a.legend(loc="lower right", fontsize=9)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # --- Panel B: ordinal axis vs ARI, per slice -------------------------------
    ari_by_sample = {r["sample"]: r["ours"]["consensus"] for r in multislice["per_slice"]}
    seen = set()
    for row in axis["per_slice"]:
        subject = row["subject"]
        x = row["mean"]
        yv = ari_by_sample[row["sample"]]
        ax_b.scatter(x, yv, color=SUBJECT_COLORS[subject], s=90, alpha=0.85,
                     label=subject if subject not in seen else None)
        seen.add(subject)
        ax_b.annotate(row["sample"], (x, yv), fontsize=7, xytext=(4, 4),
                      textcoords="offset points")

    from scipy.stats import spearmanr

    xs = [r["mean"] for r in axis["per_slice"]]
    ys = [ari_by_sample[r["sample"]] for r in axis["per_slice"]]
    rho = spearmanr(xs, ys)

    ax_b.set_xlabel("Existing model's implicit ordinal axis\n(|Spearman| embedding PC1 vs true layer order)")
    ax_b.set_ylabel("ARI (current model, consensus)")
    ax_b.set_title(
        "B. Subject 3 has the STRONGEST ordinal axis and the WORST ARI\n"
        f"(across-slice trend: Spearman {rho.statistic:.2f}, p={rho.pvalue:.2f} "
        "-- suggestive only, n=12)"
    )
    ax_b.legend(fontsize=9)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "topological_gate_result.png", dpi=150)
    plt.close(fig)
    print("Saved outputs/figures/topological_gate_result.png")


if __name__ == "__main__":
    main()
