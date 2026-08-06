"""BAAP evaluation on 10x Visium Human Breast Cancer.

Evaluates whether dynamic address-similarity gating + hop attention pooling
allows BAAP to adapt to smaller tumor domains (28-190 spots), preventing
over-smoothing and outperforming the baseline (consensus ARI 0.546).

Protocol:
  - 5 seeds (0..4)
  - mclust-equivalent + spatial refinement clustering
  - Consensus clustering over label sets
  - Compares against baseline SpatialAddressMemoryLayer (0.546) and GraphST (0.643)

Saves output to outputs/logs/baap_breast_cancer_results.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS, load_breast_cancer
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.train_baap_model import train_baap_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "baap_breast_cancer_results.json"

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
        "model": "BAAP",
        "seeds": seeds,
        "baselines": {
            "ours_SpatialAddressMemoryLayer": _BASELINE_OURS,
            "graphst": _BASELINE_GRAPHST,
        },
    }

    baap_labels, baap_aris, seed_diagnostics = [], [], []
    for seed in seeds:
        print(f"\n  --- seed {seed} ---", flush=True)
        model, trained, history = train_baap_model(
            adata.copy(),
            seed=seed,
            device=device,
            verbose=True,
            log_every=150,
        )
        embedding = trained.obsm["X_baap"]
        labels = cluster_embedding(embedding, N_REGIONS, coords=coords, refine=True)
        baap_labels.append(labels)
        ari = _ari(truth, labels, mask)
        baap_aris.append(ari)

        final = history[-1]
        seed_diagnostics.append({
            "seed": seed,
            "ari": ari,
            "n_slots_used": final["n_slots_used"],
            "usage_entropy": final["usage_entropy"],
            "effective_hop_depth": final["effective_hop_depth"],
            "hop_weights_mean": final["hop_weights_mean"],
            "collapsed": (
                final["n_slots_used"] <= 1
                or final["usage_entropy"] < 0.5 * final["max_entropy"]
            ),
        })
        print(
            f"  seed {seed}: ARI={ari:.4f}  "
            f"eff_hop={final['effective_hop_depth']:.2f}  "
            f"slots_used={final['n_slots_used']}",
            flush=True,
        )

    baap_consensus = consensus_cluster(baap_labels, N_REGIONS)
    consensus_ari = _ari(truth, baap_consensus, mask)

    results["baap"] = {
        "per_seed": baap_aris,
        "mean": float(np.mean(baap_aris)),
        "std": float(np.std(baap_aris)),
        "consensus": consensus_ari,
        "seed_diagnostics": seed_diagnostics,
    }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(5))

    print("=== BAAP Breast Cancer Evaluation ===")
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

    baap = results["baap"]
    print("\n=== Final Results ===")
    print(
        f"  BAAP:             per-seed = {baap['mean']:.4f} +/- {baap['std']:.4f}  "
        f"consensus = {baap['consensus']:.4f}"
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
        f"\n  delta consensus (BAAP - baseline): "
        f"{baap['consensus'] - _BASELINE_OURS['consensus']:+.4f}"
    )
    print(
        f"  delta consensus (BAAP - GraphST):  "
        f"{baap['consensus'] - _BASELINE_GRAPHST['consensus']:+.4f}"
    )

    print("\n  Per-seed hop diagnostics:")
    for d in baap["seed_diagnostics"]:
        hops = [f"{w:.3f}" for w in d["hop_weights_mean"]]
        print(
            f"    seed {d['seed']}: ARI={d['ari']:.4f}  "
            f"eff_hop={d['effective_hop_depth']:.2f}  "
            f"slots={d['n_slots_used']}  collapsed={d['collapsed']}"
        )
        print(f"      hop_weights: [{', '.join(hops)}]")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
