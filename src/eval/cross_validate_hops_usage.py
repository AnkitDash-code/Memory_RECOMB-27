"""Cross-validate n_hops and lambda_usage -- the two remaining hyperparameters
that were still single-slice-tuned (on 151673 alone) after memory_slots was
fixed by cross_validate_capacity.py.

Same train/CV-validation/test split as cross_validate_capacity.py, reusing its
slice partition so results stay comparable:

  - tuning slice (never reused): 151673
  - CV validation set (used to pick n_hops/lambda_usage below): 151508, 151670, 151674
  - true held-out test set (never used to pick anything): the other 8 slices

Coordinate-descent search rather than a full joint grid, to keep the number of
training runs affordable: first sweep n_hops with lambda_usage held at its
current default (0.1), then sweep lambda_usage with n_hops fixed at whatever
the first sweep selected. memory_slots stays fixed at 16 (already
cross-validated). This can miss an interaction the two hyperparameters might
have, but a full 2D grid at the same seed count would be ~3x the compute for a
refinement that has not been shown to matter here.
"""

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.eval.cross_validate_capacity import CV_VALIDATION_SLICES, TRUE_HOLDOUT_SLICES, TUNING_SLICE
from src.models.train_spatial_address import train_spatial_address_model

DEFAULT_N_HOPS = 4
DEFAULT_LAMBDA_USAGE = 0.1
N_HOPS_CANDIDATES = [1, 2, 3, 4, 6, 8]
LAMBDA_USAGE_CANDIDATES = [0.02, 0.05, 0.1, 0.2, 0.5]


def _ari(base, n_hops, lambda_usage, seed, device):
    truth = base.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    n_layers = int(truth.nunique())
    coords = base.obsm["spatial"]

    _, trained, _ = train_spatial_address_model(
        base.copy(), n_hops=n_hops, lambda_usage=lambda_usage,
        seed=seed, device=device, verbose=False,
    )
    labels = cluster_embedding(trained.obsm["X_spatial_address"], n_layers, coords=coords, refine=True)
    return adjusted_rand_score(truth[mask], np.asarray(labels)[mask])


def _cross_validate(candidates, fixed_other, is_n_hops, seeds, device, cache):
    """candidates varies one of {n_hops, lambda_usage}; fixed_other supplies the
    other, held constant. is_n_hops=True means `candidates` are n_hops values."""
    scores = {}
    for value in candidates:
        n_hops = value if is_n_hops else fixed_other
        lambda_usage = fixed_other if is_n_hops else value
        per_slice = [
            np.mean([_ari(cache[s], n_hops, lambda_usage, seed, device) for seed in seeds])
            for s in CV_VALIDATION_SLICES
        ]
        scores[value] = float(np.mean(per_slice))
        label = "n_hops" if is_n_hops else "lambda_usage"
        print(f"{label}={value:<5} CV mean={scores[value]:.4f}", flush=True)
    return max(scores, key=scores.get), scores


def evaluate_on_true_holdout(n_hops, lambda_usage, seeds, device):
    means = []
    for sample in TRUE_HOLDOUT_SLICES:
        base = preprocess_hvg(load_dlpfc_slice(sample))
        aris = [_ari(base, n_hops, lambda_usage, seed, device) for seed in seeds]
        means.append(float(np.mean(aris)))
        print(f"  {sample}: {means[-1]:.4f}", flush=True)
    return float(np.mean(means)), float(np.std(means))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [0, 1, 2]
    cache = {s: preprocess_hvg(load_dlpfc_slice(s)) for s in CV_VALIDATION_SLICES}

    print("=== Cross-validating n_hops on", CV_VALIDATION_SLICES,
          f"(lambda_usage held at default {DEFAULT_LAMBDA_USAGE}) ===")
    best_n_hops, n_hops_scores = _cross_validate(
        N_HOPS_CANDIDATES, DEFAULT_LAMBDA_USAGE, True, seeds, device, cache
    )
    print(f"\nSelected n_hops={best_n_hops} by cross-validation "
          f"(single-slice tuning on {TUNING_SLICE} alone had picked {DEFAULT_N_HOPS})\n")

    print("=== Cross-validating lambda_usage on", CV_VALIDATION_SLICES,
          f"(n_hops held at CV-selected {best_n_hops}) ===")
    best_lambda_usage, lambda_scores = _cross_validate(
        LAMBDA_USAGE_CANDIDATES, best_n_hops, False, seeds, device, cache
    )
    print(f"\nSelected lambda_usage={best_lambda_usage} by cross-validation "
          f"(single-slice tuning on {TUNING_SLICE} alone had picked {DEFAULT_LAMBDA_USAGE})\n")

    print(f"=== Evaluating n_hops={best_n_hops}, lambda_usage={best_lambda_usage} "
          f"on {len(TRUE_HOLDOUT_SLICES)} true held-out slices ===")
    mean, std = evaluate_on_true_holdout(best_n_hops, best_lambda_usage, seeds, device)
    print(f"\nCV-selected (n_hops={best_n_hops}, lambda_usage={best_lambda_usage}): {mean:.4f} +/- {std:.4f}")

    print(f"\n=== Baseline for comparison: current defaults "
          f"(n_hops={DEFAULT_N_HOPS}, lambda_usage={DEFAULT_LAMBDA_USAGE}) on the same held-out set ===")
    baseline_mean, baseline_std = evaluate_on_true_holdout(DEFAULT_N_HOPS, DEFAULT_LAMBDA_USAGE, seeds, device)
    print(f"\nDefaults (n_hops={DEFAULT_N_HOPS}, lambda_usage={DEFAULT_LAMBDA_USAGE}): "
          f"{baseline_mean:.4f} +/- {baseline_std:.4f}")


if __name__ == "__main__":
    main()
