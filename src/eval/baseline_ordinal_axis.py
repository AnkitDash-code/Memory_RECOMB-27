"""Does the EXISTING (Stage 13, non-topological) model already encode cortical
layer order implicitly?

This question was not in the TOM plan, but the plan's own gate check surfaced
it as decisive: the control condition (PC1 of the current model's embedding
vs. true layer order) scored |Spearman| = 0.846 on DLPFC 151673 -- far higher
than expected for an architecture the plan describes as having "an *unordered*
address space [where] nothing says slot 3 is 'between' slot 2 and slot 4".

If that holds across slices and seeds, it substantially weakens the TOM
premise: the contribution would not be "create an ordinal axis that does not
exist" (one already emerges, strongly) but the much narrower "sharpen an
ordinal axis that already emerges". Worth measuring properly before building
anything further on the original premise.

Reported per slice and per subject, over multiple seeds, since a single-seed
single-slice number is exactly the kind of evidence this project has twice
been burned by (Stage 14's promising single-seed entmax result, Stage 15's
promising single-seed contrastive result -- both evaporated across seeds).
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.analyze_multislice_variance import SUBJECTS
from src.eval.topological_axis_diagnostic import _layer_ordinal
from src.models.train_spatial_address import train_spatial_address_model

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "logs" / "baseline_ordinal_axis.json"
)


def run_slice(sample, seeds, device):
    adata = preprocess_hvg(load_dlpfc_slice(sample))
    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    depth = _layer_ordinal(truth.to_numpy())

    rhos = []
    for seed in seeds:
        _, trained, _ = train_spatial_address_model(
            adata.copy(), seed=seed, device=device, verbose=False
        )
        pc1 = PCA(n_components=1).fit_transform(trained.obsm["X_spatial_address"])[:, 0]
        rhos.append(abs(float(spearmanr(pc1[mask], depth[mask]).statistic)))

    return {
        "sample": sample,
        "per_seed_abs_spearman": rhos,
        "mean": float(np.mean(rhos)),
        "std": float(np.std(rhos)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [0, 1, 2]

    results = []
    print(f"{'sample':<10}{'subject':<10}{'mean |rho| (PC1 vs depth)':>28}{'std':>9}")
    print("-" * 57)
    for subject, samples in SUBJECTS.items():
        for sample in samples:
            row = run_slice(sample, seeds, device)
            row["subject"] = subject
            results.append(row)
            print(f"{sample:<10}{subject:<10}{row['mean']:>28.4f}{row['std']:>9.4f}", flush=True)

    print("\n=== per-subject ===")
    per_subject = {}
    for subject in SUBJECTS:
        rows = [r for r in results if r["subject"] == subject]
        m = float(np.mean([r["mean"] for r in rows]))
        per_subject[subject] = m
        print(f"  {subject}: {m:.4f}")

    overall = float(np.mean([r["mean"] for r in results]))
    print(f"\noverall mean |Spearman| (PC1 vs true layer order): {overall:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"per_slice": results, "per_subject": per_subject, "overall": overall}, indent=2)
    )
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
