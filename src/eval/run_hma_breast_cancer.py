"""HMA evaluation on 10x Visium Human Breast Cancer.

Evaluates whether Hopfield Memory Attractor Dynamics (snapping blurred boundary
addresses back toward clean key prototypes after deep propagation) recovers
domain boundaries and outperforms the baseline (consensus ARI 0.546) and
GraphST (consensus ARI 0.643).

Protocol:
  - 5 seeds (0..4)
  - mclust-equivalent + spatial refinement clustering
  - Consensus clustering over label sets

Saves output to outputs/logs/hma_breast_cancer_results.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS, load_breast_cancer
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.train_hma_model import train_hma_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "hma_breast_cancer_results.json"

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
        "model": "HMA",
        "seeds": seeds,
        "baselines": {
            "ours_SpatialAddressMemoryLayer": _BASELINE_OURS,
            "graphst": _BASELINE_GRAPHST,
        },
    }

    hma_labels, hma_aris, seed_diagnostics = [], [], []
    for seed in seeds:
        print(f"\n  --- seed {seed} ---", flush=True)
        model, trained, history = train_hma_model(
            adata.copy(),
            seed=seed,
            device=device,
            verbose=True,
            log_every=150,
        )
        embedding = trained.obsm["X_hma"]
        labels = cluster_embedding(embedding, N_REGIONS, coords=coords, refine=True)
        hma_labels.append(labels)
        ari = _ari(truth, labels, mask)
        hma_aris.append(ari)

        final = history[-1]
        seed_diagnostics.append({
            "seed": seed,
            "ari": ari,
            "n_slots_used": final["n_slots_used"],
            "usage_entropy": final["usage_entropy"],
            "attractor_beta": final["attractor_beta"],
            "collapsed": (
                final["n_slots_used"] <= 1
                or final["usage_entropy"] < 0.5 * final["max_entropy"]
            ),
        })
        print(
            f"  seed {seed}: ARI={ari:.4f}  "
            f"attractor_beta={final['attractor_beta']:.4f}  "
            f"slots_used={final['n_slots_used']}",
            flush=True,
        )

    hma_consensus = consensus_cluster(hma_labels, N_REGIONS)
    consensus_ari = _ari(truth, hma_consensus, mask)

    results["hma"] = {
        "per_seed": hma_aris,
        "mean": float(np.mean(hma_aris)),
        "std": float(np.std(hma_aris)),
        "consensus": consensus_ari,
        "seed_diagnostics": seed_diagnostics,
    }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(5))

    print("=== HMA Breast Cancer Evaluation ===")
    print(f"Device: {device}  |  Seeds: {seeds}")
    print(
        f"Baseline (SpatialAddressMemoryLayer):  "
        f"per-seed={_BASELINE_OURS['mean']:.3f}  "
        f"consensus={_BASELINE_OURS['consensus']:.3f}"
    )
    print(
        f"GraphST baseline:                      "
        f"per-seed={_BASELINE_GRAPHST['mean']:.3f}  "
        f"consensus={_BASELINE_GRAPHST['consensus']:.3f}"
    )

    results = evaluate(seeds, device)

    hma = results["hma"]
    print("\n=== Final Results ===")
    print(
        f"  HMA:              per-seed = {hma['mean']:.4f} +/- {hma['std']:.4f}  "
        f"consensus = {hma['consensus']:.4f}"
    )
    print(
        f"  Baseline (ours):  per-seed = {_BASELINE_OURS['mean']:.4f} +/- "
        f"{_BASELINE_OURS['std']:.4f}  "
        f"consensus = {_BASELINE_OURS['consensus']:.4f}"
    )
    print(
        f"  GraphST:          per-seed = {_BASELINE_GRAPHST['mean']:.4f} +/- "
        f"{_BASELINE_GRAPHST['std']:.4f}  "
        f"consensus = {_BASELINE_GRAPHST['consensus']:.4f}"
    )
    print(
        f"\n  delta consensus (HMA - baseline): "
        f"{hma['consensus'] - _BASELINE_OURS['consensus']:+.4f}"
    )
    print(
        f"  delta consensus (HMA - GraphST):  "
        f"{hma['consensus'] - _BASELINE_GRAPHST['consensus']:+.4f}"
    )

    print("\n  Per-seed attractor diagnostics:")
    for d in hma["seed_diagnostics"]:
        print(
            f"    seed {d['seed']}: ARI={d['ari']:.4f}  "
            f"attractor_beta={d['attractor_beta']:.4f}  "
            f"slots={d['n_slots_used']}  collapsed={d['collapsed']}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
