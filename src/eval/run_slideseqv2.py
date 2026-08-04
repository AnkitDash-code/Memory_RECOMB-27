"""Phase C: Slide-seqV2 mouse hippocampus -- unsupervised metrics only, not
a supervised ARI headline. See `src/data/load_slideseqv2.py` for why: squidpy's
distribution carries cell-type labels (not spatial domains) and no raw counts.

Both limitations affect ours and GraphST equally (same data, same missing
raw counts), so the ours-vs-GraphST comparison stays fair even though neither
absolute number is comparable to a properly-preprocessed run. K=14 is used as
a common, convenient cluster count for both methods (matching the cell-type
count) -- not a claim that 14 is the "true" number of spatial domains, which
is unknown here.

Metrics: silhouette (embedding separation) and spatial coherence (mean
Moran's I of cluster indicators, `src/eval/metrics.py`), the same unsupervised
proxies used for the Phase 0 Visium mouse-brain benchmark. A caveated ARI
against the cell-type labels is also reported, explicitly flagged as
measuring a different task (cell-type recovery, not domain identification),
never as a headline number.

3 seeds (not 5): this is a secondary, lower-priority check relative to the
breast cancer result (which had a literature ARI to compare against), so
effort is intentionally bounded, matching the Garfield "reduced check" precedent.

No consensus-across-seeds here (unlike DLPFC/breast cancer): `consensus_cluster`'s
co-association matrix is O(n^2) in memory -- fine at DLPFC/breast-cancer scale
(~3-5k spots) but a dense (41786, 41786) float64 distance matrix is ~14GB, and
even the condensed form sklearn's AgglomerativeClustering builds internally
still exceeded available memory (confirmed by a real `ArrayMemoryError`, not
assumed). Per-seed mean/std is reported instead.

**Subsampled to 12,000 spots (fixed seed), a hardware-driven necessity, not a
convenience.** GraphST's own package -- both `construct_interaction` (dense
pairwise distances) AND `construct_interaction_KNN` (its own documented
large-N alternative for `datatype in ['Stereo', 'Slide']`) -- materializes a
DENSE (n_spots, n_spots) `adj` matrix regardless of construction method;
`GraphSTModel.__init__` then unconditionally `.copy()`s the adata holding it.
At the full 41,786 spots that's a confirmed `ArrayMemoryError` (13GB for one
copy, on a 16GB-RAM machine) -- GraphST's package genuinely cannot run on
this dataset at full scale on this hardware, not something our own code can
route around, since the ceiling is inside GraphST's own construction
functions. Subsampling to 12,000 spots (~1.4GB dense matrix) is applied
identically before both methods train, so the comparison stays fair; it does
mean these numbers describe a subsample, not the full hippocampus section.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_slideseqv2 import load_slideseqv2
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.eval.metrics import embedding_silhouette, spatial_coherence
from src.models.run_graphst import run_graphst
from src.models.train_spatial_address import train_spatial_address_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "slideseqv2_results.json"
N_CLUSTERS = 14  # matches the cell-type count; NOT a claim about true domain count
SUBSAMPLE_N = 12000  # hardware ceiling on GraphST's dense adjacency, see module docstring
SUBSAMPLE_SEED = 0


def _score(adata, cluster_key, embedding_key, cell_type):
    labels = adata.obs[cluster_key].to_numpy()
    embedding = adata.obsm[embedding_key]
    return {
        "silhouette": embedding_silhouette(embedding, labels),
        "spatial_coherence_morans_i": spatial_coherence(adata, cluster_key)["mean"],
        "ari_vs_cell_type_CAVEAT": float(adjusted_rand_score(cell_type, labels)),
    }


def evaluate(seeds, device):
    raw = load_slideseqv2()
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    keep = rng.choice(raw.n_obs, size=min(SUBSAMPLE_N, raw.n_obs), replace=False)
    raw = raw[np.sort(keep)].copy()

    adata = preprocess_hvg(raw.copy(), coord_type="generic", platform="slideseqv2")
    cell_type = adata.obs["cell_type"].to_numpy()
    coords = adata.obsm["spatial"]

    results = {"n_spots": int(adata.n_obs), "n_clusters": N_CLUSTERS, "subsampled_from": 41786}

    ours_scores = []
    for seed in seeds:
        _, trained, _ = train_spatial_address_model(
            adata.copy(), seed=seed, device=device, verbose=False,
        )
        embedding = trained.obsm["X_spatial_address"]
        labels = cluster_embedding(embedding, N_CLUSTERS, coords=coords, refine=True)
        trained.obs["_pred"] = labels
        ours_scores.append(_score(trained, "_pred", "X_spatial_address", cell_type))
    results["ours"] = {
        "per_seed": ours_scores,
        "silhouette_mean": float(np.mean([s["silhouette"] for s in ours_scores])),
        "silhouette_std": float(np.std([s["silhouette"] for s in ours_scores])),
        "spatial_coherence_mean": float(np.mean([s["spatial_coherence_morans_i"] for s in ours_scores])),
        "spatial_coherence_std": float(np.std([s["spatial_coherence_morans_i"] for s in ours_scores])),
        "ari_vs_cell_type_mean_CAVEAT": float(np.mean([s["ari_vs_cell_type_CAVEAT"] for s in ours_scores])),
    }

    graphst_scores = []
    for seed in seeds:
        graphst_adata = run_graphst(
            raw.copy(), n_clusters=N_CLUSTERS, device=device,
            random_seed=seed, cluster=False, datatype="Slide",
        )
        labels = cluster_embedding(
            graphst_adata.obsm["emb"], N_CLUSTERS,
            coords=graphst_adata.obsm["spatial"], refine=True,
        )
        graphst_adata.obs["_pred"] = labels
        ct = adata.obs["cell_type"].reindex(graphst_adata.obs_names).to_numpy()
        graphst_scores.append(_score(graphst_adata, "_pred", "emb", ct))
    results["graphst"] = {
        "per_seed": graphst_scores,
        "silhouette_mean": float(np.mean([s["silhouette"] for s in graphst_scores])),
        "silhouette_std": float(np.std([s["silhouette"] for s in graphst_scores])),
        "spatial_coherence_mean": float(np.mean([s["spatial_coherence_morans_i"] for s in graphst_scores])),
        "spatial_coherence_std": float(np.std([s["spatial_coherence_morans_i"] for s in graphst_scores])),
        "ari_vs_cell_type_mean_CAVEAT": float(np.mean([s["ari_vs_cell_type_CAVEAT"] for s in graphst_scores])),
    }

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(3))

    results = evaluate(seeds, device)

    print(f"n_spots={results['n_spots']}  n_clusters={results['n_clusters']} (cell-type count, not a domain-count claim)")
    for method in ("ours", "graphst"):
        r = results[method]
        print(f"  {method:10s} silhouette={r['silhouette_mean']:.4f}  "
              f"spatial_coherence={r['spatial_coherence_mean']:.4f}  "
              f"ARI-vs-cell-type(CAVEAT, different task)={r['ari_vs_cell_type_mean_CAVEAT']:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
