"""Multi-seed check on the ONE positive signal from the TOM sweep.

The sigma sweep / ablation found every SOM-enabled configuration collapses,
but `ordinal-only` (lambda_som=0, lambda_ordinal=0.02) scored ARI 0.5696 on
151673 vs. 0.5150 for the Stage 13 baseline -- the only configuration that
looked better than what we already have.

This project has been burned twice by exactly this shape of evidence: Stage
14's entmax and Stage 15's contrastive loss both produced promising
single-seed numbers that vanished across 5 seeds. So this gets a multi-seed,
multi-slice check before it is reported as anything.

IMPORTANT FRAMING, independent of the outcome: even if this holds up, it is
NOT the TOM hypothesis. With lambda_som=0 there is no SOM term, so the slot
ordering is arbitrary (measured: |Spearman| of the ordinal axis vs. true
cortical depth is 0.151, essentially the same as the 0.175 obtained with no
topology terms at all). Any benefit would therefore come from a generic
extra spatial-smoothness regularizer applied through a random 1D projection
of the address simplex -- overlapping in purpose with the n_hops address
propagation already in the model -- not from a meaningful laminar ordering.
It would be a mundane finding wearing the plan's clothes, and must be
reported that way.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.models.train_topological import train_topological_model

RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "ordinal_only_check.json"

# One slice per subject plus the slice where the effect was first seen.
SAMPLES = ["151507", "151669", "151673", "151674"]
SEEDS = [0, 1, 2, 3, 4]


def run_config(adata, truth, mask, n_layers, coords, lambda_ordinal, seed, device):
    _, trained, _ = train_topological_model(
        adata.copy(), seed=seed, device=device, verbose=False,
        lambda_som=0.0, lambda_ordinal=lambda_ordinal,
    )
    labels = cluster_embedding(
        trained.obsm["X_topological_address"], n_layers, coords=coords, refine=True
    )
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    print(f"{'sample':<10}{'baseline (mean+/-std)':>26}{'ordinal-only (mean+/-std)':>28}{'delta':>9}")
    print("-" * 73)
    for sample in SAMPLES:
        adata = preprocess_hvg(load_dlpfc_slice(sample))
        truth = adata.obs["ground_truth_layer"]
        mask = truth.notna().to_numpy()
        n_layers = int(truth.nunique())
        coords = adata.obsm["spatial"]

        base = [run_config(adata, truth, mask, n_layers, coords, 0.0, s, device) for s in SEEDS]
        ordi = [run_config(adata, truth, mask, n_layers, coords, 0.02, s, device) for s in SEEDS]

        row = {
            "sample": sample,
            "baseline_per_seed": base,
            "baseline_mean": float(np.mean(base)),
            "baseline_std": float(np.std(base)),
            "ordinal_per_seed": ordi,
            "ordinal_mean": float(np.mean(ordi)),
            "ordinal_std": float(np.std(ordi)),
            "delta": float(np.mean(ordi) - np.mean(base)),
        }
        results.append(row)
        print(f"{sample:<10}{row['baseline_mean']:>18.4f} +/-{row['baseline_std']:<6.4f}"
              f"{row['ordinal_mean']:>20.4f} +/-{row['ordinal_std']:<6.4f}"
              f"{row['delta']:>9.4f}", flush=True)

    mean_delta = float(np.mean([r["delta"] for r in results]))
    wins = sum(1 for r in results if r["delta"] > 0)
    print(f"\nmean delta = {mean_delta:+.4f}   ordinal-only better on {wins}/{len(results)} slices")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"per_slice": results, "mean_delta": mean_delta, "wins": wins}, indent=2)
    )
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
