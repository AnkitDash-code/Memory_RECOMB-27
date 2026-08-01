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


def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation (Kerby 2014), the standard
    effect size for a Wilcoxon signed-rank test: (W+ - W-) / (W+ + W-).

    Range [-1, 1]; positive means `ours` above `graphst`. Reported because at
    n=11 "not significant" and "no effect" are different claims, and a p-value
    alone cannot distinguish an underpowered test from a genuinely null one.
    """
    from scipy.stats import rankdata

    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total else 0.0


def bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean paired difference, resampling
    SLICES with replacement.

    At n=11 a single p-value is a thin summary; an interval estimate conveys
    the range of gaps the data is actually consistent with, which is what a
    reader needs in order to judge whether "no significant difference" means
    "close" or merely "too few slices to tell".
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run(field, label):
    samples, ours, graphst = _paired_arrays(field)
    diff = ours - graphst

    w_stat, w_p = wilcoxon(ours, graphst)
    t_stat, t_p = ttest_rel(ours, graphst)
    sh_stat, sh_p = shapiro(diff)
    effect = rank_biserial(diff)
    ci_lo, ci_hi = bootstrap_ci(diff)

    print(f"=== {label} ({field}) ===")
    print(f"  n slices = {len(samples)}, ours wins on {(diff > 0).sum()}/{len(samples)}")
    print(f"  mean diff (ours - graphst) = {diff.mean():.4f}, std = {diff.std():.4f}")
    print(f"  bootstrap 95% CI on mean diff: [{ci_lo:+.4f}, {ci_hi:+.4f}]  (10,000 resamples)")
    print(f"  rank-biserial effect size: {effect:+.4f}")
    print(f"  Shapiro-Wilk on diff (normality): stat={sh_stat:.4f}, p={sh_p:.4f} "
          f"({'looks normal' if sh_p > 0.05 else 'NOT normal -- trust Wilcoxon over t-test'})")
    print(f"  Wilcoxon signed-rank: stat={w_stat:.1f}, p={w_p:.4f}")
    print(f"  Paired t-test:        stat={t_stat:.4f}, p={t_p:.4f}")
    print(f"  {'SIGNIFICANT at alpha=0.05' if w_p < 0.05 else 'NOT significant at alpha=0.05'} "
          f"(Wilcoxon)")
    print(f"  CI {'excludes' if (ci_lo > 0 or ci_hi < 0) else 'includes'} zero"
          f" -- {'consistent with a real difference' if (ci_lo > 0 or ci_hi < 0) else 'data is consistent with no difference, but also with gaps up to ' + f'{max(abs(ci_lo), abs(ci_hi)):.3f}'}")
    print()
    return {"field": field, "n": len(samples), "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std()), "wilcoxon_p": float(w_p), "ttest_p": float(t_p),
            "shapiro_p": float(sh_p), "rank_biserial": effect,
            "bootstrap_ci_lo": ci_lo, "bootstrap_ci_hi": ci_hi}


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
