"""Three-way paired significance test: ours vs. GraphST vs. STAGATE.

STAGATE was recorded as "blocked on Windows" in this project; that claim was
stale (PyG's wheel index now covers torch 2.11.0) and it runs locally --
src/eval/run_stagate_dlpfc.py, same 12-slice x 5-seed x shared-clustering
protocol as every other method here. This adds it as a second real comparator
alongside GraphST, using the same paired-test machinery (Wilcoxon, rank-
biserial effect size, bootstrap CI) already validated in significance_test.py,
since a single comparator makes any "no significant difference" claim weaker
than it needs to be.

Three pairwise comparisons, not a single omnibus test: with only n=11 slices,
a 3-group omnibus test (e.g. Friedman) would have even less power than the
pairwise Wilcoxon tests already reported, and the question that matters is
specifically "how do we compare to EACH established method", not "are the
three methods different from each other in general".
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon

from src.eval.significance_test import bootstrap_ci, rank_biserial

MAIN_RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "dlpfc_multislice_results.json"
STAGATE_RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "stagate_dlpfc_results.json"
TUNING_SLICE = "151673"


def _load(field):
    main = json.loads(MAIN_RESULTS_PATH.read_text())["per_slice"]
    stagate = json.loads(STAGATE_RESULTS_PATH.read_text())["per_slice"]
    main = {r["sample"]: r for r in main if r["sample"] != TUNING_SLICE}
    stagate = {r["sample"]: r for r in stagate if r["sample"] != TUNING_SLICE}
    samples = sorted(set(main) & set(stagate))
    ours = np.array([main[s]["ours"][field] for s in samples])
    graphst = np.array([main[s]["graphst"][field] for s in samples])
    stag = np.array([stagate[s][field] for s in samples])
    return samples, ours, graphst, stag


def compare(name_a, a, name_b, b, field, label):
    diff = a - b
    w_stat, w_p = wilcoxon(a, b)
    t_stat, t_p = ttest_rel(a, b)
    sh_stat, sh_p = shapiro(diff)
    effect = rank_biserial(diff)
    ci_lo, ci_hi = bootstrap_ci(diff)

    print(f"--- {name_a} vs {name_b} ({label}, {field}) ---")
    print(f"  n={len(a)}, {name_a} wins {(diff > 0).sum()}/{len(a)}")
    print(f"  mean diff ({name_a}-{name_b}) = {diff.mean():+.4f}, "
          f"bootstrap 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  rank-biserial = {effect:+.4f}   Wilcoxon p = {w_p:.4f}   "
          f"(Shapiro p={sh_p:.4f}, {'normal' if sh_p > 0.05 else 'non-normal, trust Wilcoxon'})")
    print(f"  {'SIGNIFICANT' if w_p < 0.05 else 'not significant'} at alpha=0.05")
    print()
    return {"pair": f"{name_a}_vs_{name_b}", "field": field, "mean_diff": float(diff.mean()),
            "bootstrap_ci": [ci_lo, ci_hi], "rank_biserial": effect,
            "wilcoxon_p": float(w_p), "ttest_p": float(t_p), "shapiro_p": float(sh_p)}


def main():
    results = []
    for field, label in [("mean", "per-seed"), ("consensus", "consensus")]:
        samples, ours, graphst, stagate = _load(field)
        print(f"=== {label} metric, n={len(samples)} held-out slices "
              f"(both GraphST and STAGATE available) ===\n")
        results.append(compare("ours", ours, "graphst", graphst, field, label))
        results.append(compare("ours", ours, "stagate", stagate, field, label))
        results.append(compare("graphst", graphst, "stagate", stagate, field, label))

    out_path = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "significance_test_stagate.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
