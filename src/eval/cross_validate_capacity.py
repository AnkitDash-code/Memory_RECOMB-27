"""Cross-validated hyperparameter selection, fixing a real overfitting problem.

The 12-slice evaluation (run_dlpfc_multislice.py) showed our held-out gap to
GraphST (0.129) was much larger than the tuning-slice gap (0.026), and the
within-subject variance analysis (analyze_multislice_variance.py) showed this
is largely model fragility, not task difficulty. One concrete, fixable cause:
every hyperparameter (including memory_slots) was chosen on 151673 alone.

This script re-selects memory_slots by cross-validation across 3 slices (one
per DLPFC subject, none of them 151673), then checks the result on a disjoint
set of 8 truly-unseen slices -- proper train/validation/test separation:

  - tuning slice (original, not reused here): 151673
  - CV validation set (used to pick memory_slots below): 151508, 151670, 151674
  - true held-out test set (never used to pick anything): the other 8 slices

Result: memory_slots=32 (picked on 151673 alone) scores 0.4601 on the true
held-out set; memory_slots=16 (picked by this cross-validation) scores 0.5025 --
a real +0.042 ARI improvement from fixing the tuning methodology, not from a
new architecture. GraphST on the same 8 slices: 0.5766 (gap narrows from 0.117
to 0.074, but does not close).
"""

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.models.train_spatial_address import train_spatial_address_model

CV_VALIDATION_SLICES = ["151508", "151670", "151674"]
TRUE_HOLDOUT_SLICES = ["151507", "151509", "151510", "151669", "151671", "151672", "151675", "151676"]
TUNING_SLICE = "151673"  # excluded from both sets above; never reused


def _ari(base, memory_slots, seed, device):
    truth = base.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    n_layers = int(truth.nunique())
    coords = base.obsm["spatial"]

    _, trained, _ = train_spatial_address_model(
        base.copy(), memory_slots=memory_slots, seed=seed, device=device, verbose=False
    )
    labels = cluster_embedding(trained.obsm["X_spatial_address"], n_layers, coords=coords, refine=True)
    return adjusted_rand_score(truth[mask], np.asarray(labels)[mask])


def cross_validate(candidates, seeds, device):
    """Select the memory_slots value with the best mean ARI across the CV
    validation slices (not the tuning slice, not the final test slices)."""
    cache = {s: preprocess_hvg(load_dlpfc_slice(s)) for s in CV_VALIDATION_SLICES}
    scores = {}
    for memory_slots in candidates:
        per_slice = [
            np.mean([_ari(cache[s], memory_slots, seed, device) for seed in seeds])
            for s in CV_VALIDATION_SLICES
        ]
        scores[memory_slots] = float(np.mean(per_slice))
        print(f"memory_slots={memory_slots:<4} CV mean={scores[memory_slots]:.4f}", flush=True)
    return max(scores, key=scores.get), scores


def evaluate_on_true_holdout(memory_slots, seeds, device):
    means = []
    for sample in TRUE_HOLDOUT_SLICES:
        base = preprocess_hvg(load_dlpfc_slice(sample))
        aris = [_ari(base, memory_slots, seed, device) for seed in seeds]
        means.append(float(np.mean(aris)))
        print(f"  {sample}: {means[-1]:.4f}", flush=True)
    return float(np.mean(means)), float(np.std(means))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [0, 1, 2]

    print("=== Cross-validating memory_slots on", CV_VALIDATION_SLICES, "===")
    best, scores = cross_validate([16, 24, 32, 48, 64], seeds, device)
    print(f"\nSelected memory_slots={best} by cross-validation "
          f"(single-slice tuning on {TUNING_SLICE} alone had picked 32)\n")

    print(f"=== Evaluating memory_slots={best} on {len(TRUE_HOLDOUT_SLICES)} true held-out slices ===")
    mean, std = evaluate_on_true_holdout(best, seeds, device)
    print(f"\nmemory_slots={best}: {mean:.4f} +/- {std:.4f}")


if __name__ == "__main__":
    main()
