"""AGAP evaluation on 10x Visium Human Breast Cancer.

Evaluates whether a learned per-spot adaptive propagation-depth gate lets
AGAP adapt to smaller tumor domains (28-190 spots), preventing over-smoothing
and outperforming the baseline (consensus ARI 0.546). This is the same
adaptive-gate idea already rejected once in this project (Phase D,
`adaptive_hops` on SpatialAddressMemoryLayer -- collapsed to depth 0 without
regularization, underperformed even with lambda_hop_usage); AGAP is a
distinct implementation of the same mechanism family, worth its own
real result rather than assuming the Phase D verdict transfers.

Never run before this pass -- outputs/logs/agap_breast_cancer_results.json
did not exist prior. Protocol matches BAAP/HMA/MSAP's legacy (non-block)
comparators exactly, so the numbers are directly comparable to those.

Saves output to outputs/logs/agap_breast_cancer_results.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS, load_breast_cancer
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.train_agap_model import train_agap_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "agap_breast_cancer_results.json"

_BASELINE_OURS = {"mean": 0.412, "std": 0.072, "consensus": 0.546}
_BASELINE_GRAPHST = {"mean": 0.621, "std": 0.021, "consensus": 0.643}


def _ari(truth, labels, mask):
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def evaluate(seeds, device):
    raw = load_breast_cancer()
    adata = preprocess_hvg(raw.copy(), platform="visium")

    truth = adata.obs["ground_truth_region"]
    mask = truth.notna().to_numpy()
    coords = adata.obsm["spatial"]

    results = {
        "n_spots": int(adata.n_obs),
        "n_regions": N_REGIONS,
        "model": "AGAP",
        "seeds": seeds,
        "baselines": {
            "ours_SpatialAddressMemoryLayer": _BASELINE_OURS,
            "graphst": _BASELINE_GRAPHST,
        },
    }

    labels_per_seed, aris, seed_diagnostics = [], [], []
    for seed in seeds:
        print(f"\n  --- seed {seed} ---", flush=True)
        model, trained, history = train_agap_model(
            adata.copy(), seed=seed, device=device, verbose=True, log_every=150,
        )
        embedding = trained.obsm["X_agap"]
        labels = cluster_embedding(embedding, N_REGIONS, coords=coords, refine=True)
        labels_per_seed.append(labels)
        ari = _ari(truth, labels, mask)
        aris.append(ari)

        final = history[-1]
        collapsed = (
            final["n_slots_used"] <= 1
            or final["usage_entropy"] < 0.5 * final["max_entropy"]
        )
        seed_diagnostics.append({
            "seed": seed,
            "ari": ari,
            "n_slots_used": final["n_slots_used"],
            "usage_entropy": final["usage_entropy"],
            "median_row_entropy": final["median_row_entropy"],
            "collapsed": collapsed,
        })
        print(
            f"  seed {seed}: ARI={ari:.4f}  slots_used={final['n_slots_used']}  "
            f"usage_entropy={final['usage_entropy']:.3f}/{final['max_entropy']:.3f}",
            flush=True,
        )

    consensus_labels = consensus_cluster(labels_per_seed, N_REGIONS)
    consensus_ari = _ari(truth, consensus_labels, mask)

    results["agap"] = {
        "per_seed": aris,
        "mean": float(np.mean(aris)),
        "std": float(np.std(aris)),
        "consensus": consensus_ari,
        "seed_diagnostics": seed_diagnostics,
    }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(5))

    print("=== AGAP Breast Cancer Evaluation ===")
    print(f"Device: {device}  |  Seeds: {seeds}")

    results = evaluate(seeds, device)

    agap = results["agap"]
    print("\n=== Final Results ===")
    print(f"  AGAP:             per-seed = {agap['mean']:.4f} +/- {agap['std']:.4f}  consensus = {agap['consensus']:.4f}")
    print(f"  Baseline (ours):  per-seed = {_BASELINE_OURS['mean']:.4f} +/- {_BASELINE_OURS['std']:.4f}  consensus = {_BASELINE_OURS['consensus']:.4f}")
    print(f"  GraphST:          per-seed = {_BASELINE_GRAPHST['mean']:.4f} +/- {_BASELINE_GRAPHST['std']:.4f}  consensus = {_BASELINE_GRAPHST['consensus']:.4f}")
    print(f"\n  delta consensus (AGAP - baseline): {agap['consensus'] - _BASELINE_OURS['consensus']:+.4f}")
    print(f"  delta consensus (AGAP - GraphST):  {agap['consensus'] - _BASELINE_GRAPHST['consensus']:+.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
