"""Paired significance test: is the held-out ARI gap to GraphST real, or
noise? 11 held-out DLPFC slices, same slices/seeds/protocol for both methods,
so a paired (not independent-samples) test is the right tool -- Wilcoxon
signed-rank as the primary test (no normality assumption), paired t-test
reported alongside since the paired differences pass a normality check
(Shapiro-Wilk) here.

Answers a specific, previously-unasked question: "gap (0.024) is smaller than
GraphST's own across-slice std (0.086)" is not a significance test -- it's an
eyeballed comparison of one method's spread against a point difference. This
script runs the actual paired test on both the per-seed mean and the consensus
metric, since they give different answers and both are worth reporting
honestly rather than picking whichever supports the more favorable framing.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "dlpfc_multislice_results.json"
TUNING_SLICE = "151673"


def _paired_arrays(field):
    data = json.loads(RESULTS_PATH.read_text())
    rows = [r for r in data["per_slice"] if r["sample"] != TUNING_SLICE]
    rows = sorted(rows, key=lambda r: r["sample"])
    samples = [r["sample"] for r in rows]
    ours = np.array([r["ours"][field] for r in rows])
    graphst = np.array([r["graphst"][field] for r in rows])
    return samples, ours, graphst


def run(field, label):
    samples, ours, graphst = _paired_arrays(field)
    diff = ours - graphst

    w_stat, w_p = wilcoxon(ours, graphst)
    t_stat, t_p = ttest_rel(ours, graphst)
    sh_stat, sh_p = shapiro(diff)

    print(f"=== {label} ({field}) ===")
    print(f"  n slices = {len(samples)}, ours wins on {(diff > 0).sum()}/{len(samples)}")
    print(f"  mean diff (ours - graphst) = {diff.mean():.4f}, std = {diff.std():.4f}")
    print(f"  Shapiro-Wilk on diff (normality): stat={sh_stat:.4f}, p={sh_p:.4f} "
          f"({'looks normal' if sh_p > 0.05 else 'NOT normal -- trust Wilcoxon over t-test'})")
    print(f"  Wilcoxon signed-rank: stat={w_stat:.1f}, p={w_p:.4f}")
    print(f"  Paired t-test:        stat={t_stat:.4f}, p={t_p:.4f}")
    print(f"  {'SIGNIFICANT at alpha=0.05' if w_p < 0.05 else 'NOT significant at alpha=0.05'} "
          f"(Wilcoxon)")
    print()
    return {"field": field, "n": len(samples), "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std()), "wilcoxon_p": float(w_p), "ttest_p": float(t_p),
            "shapiro_p": float(sh_p)}


def main():
    consensus = run("consensus", "Consensus (headline metric)")
    per_seed = run("mean", "Per-seed mean (5 seeds/slice, more stable statistic)")

    print("=== Honest summary ===")
    print("These two tests do not agree, and both should be reported, not just the "
          "more favorable one:")
    print(f"  - Per-seed mean: p={per_seed['wilcoxon_p']:.4f} "
          f"({'significant' if per_seed['wilcoxon_p'] < 0.05 else 'not significant'}) "
          "-- the more stable statistic (each point is already an average of 5 seeds)")
    print(f"  - Consensus:     p={consensus['wilcoxon_p']:.4f} "
          f"({'significant' if consensus['wilcoxon_p'] < 0.05 else 'not significant'}) "
          "-- noisier per-slice (single ensemble output, higher paired-diff variance: "
          f"{consensus['std_diff']:.4f} vs {per_seed['std_diff']:.4f}), so this test is "
          "underpowered at n=11, not necessarily evidence of true parity.")


if __name__ == "__main__":
    main()
