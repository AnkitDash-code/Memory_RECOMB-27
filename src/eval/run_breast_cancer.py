"""Phase C: human breast cancer (10x Visium), the platform GraphST's own paper
reports a directly comparable ARI on (0.54-0.57 vs. 20-region pathologist
annotation) -- see `src/data/load_breast_cancer.py` for how this exact dataset
and annotation were traced and verified.

Single dataset, no multi-slice held-out split (unlike DLPFC's 12 slices) --
so this reports mean +/- std over 5 seeds plus consensus-across-seeds, the
same protocol used for DLPFC's single-slice ablation table, not the
held-out-slices table. The DLPFC-validated architecture defaults
(`train_spatial_address_model`'s defaults: memory_slots=16, n_hops=4,
lambda_usage=0.02, expression-weighted adjacency) are used as-is -- retuning
per dataset here would be the same leakage as tuning on the test set, just
spread across datasets instead of slices (the generalization plan's own
guardrail).
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS, load_breast_cancer
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.run_graphst import run_graphst
from src.models.train_spatial_address import train_spatial_address_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "breast_cancer_results.json"


def _ari(truth, labels, mask):
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def evaluate(seeds, device):
    raw = load_breast_cancer()
    adata = preprocess_hvg(raw.copy())

    truth = adata.obs["ground_truth_region"]
    mask = truth.notna().to_numpy()
    coords = adata.obsm["spatial"]

    results = {"n_spots": int(adata.n_obs), "n_regions": N_REGIONS}

    ours_labels, ours_aris = [], []
    for seed in seeds:
        _, trained, _ = train_spatial_address_model(
            adata.copy(), seed=seed, device=device, verbose=False,
        )
        embedding = trained.obsm["X_spatial_address"]
        labels = cluster_embedding(embedding, N_REGIONS, coords=coords, refine=True)
        ours_labels.append(labels)
        ours_aris.append(_ari(truth, labels, mask))
    ours_consensus = consensus_cluster(ours_labels, N_REGIONS)
    results["ours"] = {
        "per_seed": ours_aris,
        "mean": float(np.mean(ours_aris)),
        "std": float(np.std(ours_aris)),
        "consensus": _ari(truth, ours_consensus, mask),
    }

    graphst_labels, graphst_aris = [], []
    for seed in seeds:
        graphst_adata = run_graphst(
            raw.copy(), n_clusters=N_REGIONS, device=device,
            random_seed=seed, cluster=False,
        )
        labels = cluster_embedding(
            graphst_adata.obsm["emb"], N_REGIONS,
            coords=graphst_adata.obsm["spatial"], refine=True,
        )
        gt = truth.reindex(graphst_adata.obs_names)
        gmask = gt.notna().to_numpy()
        graphst_labels.append(labels)
        graphst_aris.append(_ari(gt, labels, gmask))
    graphst_consensus = consensus_cluster(graphst_labels, N_REGIONS)
    results["graphst"] = {
        "per_seed": graphst_aris,
        "mean": float(np.mean(graphst_aris)),
        "std": float(np.std(graphst_aris)),
        "consensus": _ari(gt, graphst_consensus, gmask),
    }

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(5))

    results = evaluate(seeds, device)

    print(f"n_spots={results['n_spots']}  n_regions={results['n_regions']}")
    for method in ("ours", "graphst"):
        r = results[method]
        print(f"  {method:10s} per-seed = {r['mean']:.4f} +/- {r['std']:.4f}  consensus = {r['consensus']:.4f}")
    print("  literature (GraphST paper, PMC9977836): 0.54-0.57")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
