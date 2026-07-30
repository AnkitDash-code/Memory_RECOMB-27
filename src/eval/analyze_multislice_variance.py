"""Why is the held-out gap (0.129) so much larger than the tuning-slice gap (0.026)?

Breaks down outputs/logs/dlpfc_multislice_results.json by DLPFC subject (3
subjects x 4 sections each) to separate two competing explanations:

  (a) some biological samples are just harder than others (between-subject effect)
  (b) our model is fragile to section-to-section variation even within one
      tissue block (within-subject effect, i.e. genuine model instability)

Finding: within-subject std for our method (~0.069) is nearly as large as the
overall across-slice std (~0.092), and clearly larger than GraphST's
within-subject std for 2 of 3 subjects (0.036, 0.025 vs. our 0.068, 0.085).
151673 (the tuning slice) is not simply "an easy subject" -- its same-subject,
same-layer-count siblings (151674-151676) score 0.35-0.40, close to the worst
slices in the whole set. This points at (b): real model fragility to
section-level nuisance variation, not dataset difficulty, as the dominant
factor the tuning-slice gap missed.
"""

import json
from pathlib import Path

import numpy as np

RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "dlpfc_multislice_results.json"

SUBJECTS = {
    "subject1": ["151507", "151508", "151509", "151510"],
    "subject2": ["151669", "151670", "151671", "151672"],
    "subject3": ["151673", "151674", "151675", "151676"],
}


def main():
    data = json.load(open(RESULTS_PATH))
    rows = {r["sample"]: r for r in data["per_slice"]}

    print(f"{'sample':<10}{'n_spots':>9}{'n_layers':>9}{'ours':>9}{'graphst':>9}{'gap':>9}")
    for r in data["per_slice"]:
        gap = r["graphst"]["mean"] - r["ours"]["mean"]
        print(f"{r['sample']:<10}{r['n_spots']:>9}{r['n_layers']:>9}"
              f"{r['ours']['mean']:>9.4f}{r['graphst']['mean']:>9.4f}{gap:>9.4f}")

    print("\n=== within-subject vs. between-subject variability ===")
    for method in ("ours", "graphst"):
        print(f"\n{method}:")
        subject_means = []
        within_stds = []
        for subject, samples in SUBJECTS.items():
            means = [rows[s][method]["mean"] for s in samples]
            subject_means.append(np.mean(means))
            within_stds.append(np.std(means))
            print(f"  {subject}: {[round(m, 3) for m in means]}  "
                  f"range={max(means) - min(means):.3f}  std={np.std(means):.3f}")
        print(f"  avg within-subject std: {np.mean(within_stds):.4f}")
        print(f"  between-subject std (of subject means): {np.std(subject_means):.4f}")

    print("\n=== n_layers effect ===")
    for n_layers in sorted({r["n_layers"] for r in data["per_slice"]}):
        means = [r["ours"]["mean"] for r in data["per_slice"] if r["n_layers"] == n_layers]
        print(f"  n_layers={n_layers}: mean(ours)={np.mean(means):.4f}  n_slices={len(means)}")

    spots = [r["n_spots"] for r in data["per_slice"]]
    ours = [r["ours"]["mean"] for r in data["per_slice"]]
    print(f"\ncorrelation(n_spots, our ARI) = {np.corrcoef(spots, ours)[0, 1]:.3f} (weak -- slice size is not the driver)")


if __name__ == "__main__":
    main()
