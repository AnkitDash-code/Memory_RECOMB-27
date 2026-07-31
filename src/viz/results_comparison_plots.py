"""Summary comparison figures for README.md: how the held-out ARI progressed
across stages, and the final per-slice ours-vs-GraphST breakdown.

Reads only already-computed, already-logged JSON results (nothing is
re-trained here) -- this script is pure visualization of numbers reported
elsewhere in outputs/logs/*.md.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def _load(name):
    return json.loads((LOGS_DIR / name).read_text())


def plot_progression():
    """Held-out (11-slice) ARI at each stage of this project's tuning history,
    ours vs. the constant GraphST baseline (unaffected by our hyperparameters,
    same seeds throughout)."""
    memslots32 = _load("dlpfc_multislice_results_memslots32.json")["summary"]["held_out_11_slices"]
    lambda01 = _load("dlpfc_multislice_results_lambda01.json")["summary"]["held_out_11_slices"]
    uniform_adj = _load("dlpfc_multislice_results_uniform_adjacency.json")["summary"]["held_out_11_slices"]
    current = _load("dlpfc_multislice_results.json")["summary"]["held_out_11_slices"]

    stages = [
        ("memory_slots=32\n(single-slice-tuned)", memslots32["ours"]),
        ("memory_slots=16\n(CV capacity), per-seed", lambda01["ours"]),
        ("+ consensus\nclustering", lambda01["ours_consensus"]),
        ("+ CV lambda_usage\n(0.1→0.02), per-seed", uniform_adj["ours"]),
        ("+ consensus", uniform_adj["ours_consensus"]),
        ("+ expr-weighted\nadjacency, per-seed", current["ours"]),
        ("+ consensus\n(current)", current["ours_consensus"]),
    ]
    labels = [s[0] for s in stages]
    means = [s[1]["mean"] for s in stages]
    stds = [s[1]["std"] for s in stages]
    graphst_mean = current["graphst_consensus"]["mean"]
    graphst_std = current["graphst_consensus"]["std"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(labels))
    cmap = plt.get_cmap("Blues")
    colors = ["#a6a6a6"] + [cmap(0.35 + 0.55 * i / max(len(labels) - 2, 1)) for i in range(len(labels) - 1)]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(graphst_mean, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"GraphST, consensus (current) = {graphst_mean:.3f} ± {graphst_std:.3f}")
    ax.fill_between([-0.5, len(labels) - 0.5], graphst_mean - graphst_std, graphst_mean + graphst_std,
                     color="#c0392b", alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Held-out ARI (11 DLPFC slices, mean ± std)")
    ax.set_title("Held-out ARI across successive, evidence-based fixes")
    ax.set_ylim(0, max(means) + max(stds) + 0.12)
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ari_progression.png", dpi=150)
    plt.close(fig)
    print("Saved outputs/figures/ari_progression.png")


def plot_per_slice_comparison():
    """Grouped bar chart: consensus ARI per slice, ours vs. GraphST, current config."""
    data = _load("dlpfc_multislice_results.json")["per_slice"]
    data = sorted(data, key=lambda r: r["sample"])
    samples = [r["sample"] for r in data]
    ours = [r["ours"]["consensus"] for r in data]
    graphst = [r["graphst"]["consensus"] for r in data]
    tuning_slice = "151673"

    x = np.arange(len(samples))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars_ours = ax.bar(x - width / 2, ours, width, label="Ours (consensus)", color="#2c5f8a")
    bars_graphst = ax.bar(x + width / 2, graphst, width, label="GraphST (consensus)", color="#c0392b")

    for i, sample in enumerate(samples):
        if sample == tuning_slice:
            ax.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylabel("ARI vs. ground truth (consensus across 5 seeds)")
    ax.set_title("Per-slice ARI, ours vs. GraphST (shaded = tuning slice 151673, excluded from headline mean)")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_slice_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved outputs/figures/per_slice_comparison.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_progression()
    plot_per_slice_comparison()


if __name__ == "__main__":
    main()
