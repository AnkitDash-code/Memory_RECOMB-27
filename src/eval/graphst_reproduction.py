"""Phase B1: why does our GraphST reproduction score 0.597 +/- 0.012 on 151673
when the literature reports ~0.633?

This matters more than a 0.036 cosmetic discrepancy. Every "no significant
difference from GraphST" claim in this repo is measured against OUR GraphST
run. If our reproduction is systematically handicapped, the headline claim is
measured against a weakened baseline -- and the same handicap would silently
reappear on every new dataset in Phase C. So it has to be explained or closed
before cross-platform work starts.

Already ruled out by inspection (not worth compute):
  * Model config. `run_graphst.py` passes GraphST's own published defaults
    verbatim -- dim_input=3000, dim_output=64, epochs=600, random_seed=41,
    lr=0.001, alpha=10/beta=1/theta=0.1/lamda1=10/lamda2=1 -- and calls
    GraphST's own preprocess/construct_interaction/add_contrastive_label/
    get_feature, so HVG count, normalization and graph construction are
    GraphST's, not ours.

Remaining suspect, tested here: the CLUSTERING step. Our `mclust_equivalent`
reproduces mclust's `EEE` model form (tied covariance) but not its
INITIALIZATION. R's mclust initializes from model-based hierarchical
agglomeration; sklearn's GaussianMixture defaults to k-means. With a
multimodal likelihood surface this is a well-known source of divergence even
when the model family matches exactly.

Sweeps the clustering knobs on a FIXED GraphST embedding (so any difference is
attributable to clustering alone, not to retraining variance): PCA dimension,
GMM initialization strategy, covariance type, and spatial refinement.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from src.data.load_dlpfc import load_dlpfc_slice
from src.eval.clustering import refine_labels_spatial
from src.models.run_graphst import run_graphst

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "logs" / "graphst_reproduction.json"
)
LITERATURE_ARI = 0.633
SAMPLE = "151673"


def _hierarchical_means(reduced, n_clusters):
    """Ward-agglomerative cluster centroids, as a stand-in for mclust's
    model-based hierarchical initialization."""
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(reduced)
    return np.stack([reduced[labels == k].mean(axis=0) for k in range(n_clusters)])


def cluster_variant(embedding, n_clusters, coords, n_pcs, init, covariance, refine, seed=2020):
    reduced = PCA(n_components=min(n_pcs, embedding.shape[1]), random_state=seed).fit_transform(embedding)

    kwargs = dict(n_components=n_clusters, covariance_type=covariance,
                  random_state=seed, n_init=10)
    if init == "hierarchical":
        kwargs["means_init"] = _hierarchical_means(reduced, n_clusters)
        kwargs["n_init"] = 1  # means_init makes repeated restarts meaningless
    else:
        kwargs["init_params"] = init

    labels = GaussianMixture(**kwargs).fit_predict(reduced).astype(str)
    if refine:
        labels = refine_labels_spatial(labels, coords, n_neighbors=50)
    return labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_dlpfc_slice(SAMPLE)
    truth = raw.obs["ground_truth_layer"]
    n_layers = int(truth.nunique())

    print(f"Training GraphST on {SAMPLE} with its published defaults (seed 41)...", flush=True)
    graphst_adata = run_graphst(raw.copy(), n_clusters=n_layers, device=device,
                                random_seed=41, cluster=False)
    embedding = graphst_adata.obsm["emb"]
    coords = graphst_adata.obsm["spatial"]
    gt = truth.reindex(graphst_adata.obs_names)
    mask = gt.notna().to_numpy()
    gt_np = gt.to_numpy()

    results = []
    header = f"{'n_pcs':>6}{'init':>18}{'cov':>8}{'refine':>8}{'ARI':>9}{'vs lit':>9}"
    print("\n" + header)
    print("-" * len(header))
    for n_pcs in [15, 20, 30, 50]:
        for init in ["kmeans", "k-means++", "random_from_data", "hierarchical"]:
            for covariance in ["tied", "full"]:
                for refine in [True, False]:
                    labels = cluster_variant(embedding, n_layers, coords,
                                             n_pcs, init, covariance, refine)
                    ari = float(adjusted_rand_score(gt_np[mask], np.asarray(labels)[mask]))
                    results.append({"n_pcs": n_pcs, "init": init, "covariance": covariance,
                                    "refine": refine, "ari": ari})
                    print(f"{n_pcs:>6}{init:>18}{covariance:>8}{str(refine):>8}"
                          f"{ari:>9.4f}{ari - LITERATURE_ARI:>+9.4f}", flush=True)

    best = max(results, key=lambda r: r["ari"])
    current = next(r for r in results
                   if r["n_pcs"] == 20 and r["init"] == "kmeans"
                   and r["covariance"] == "tied" and r["refine"])

    print(f"\ncurrent protocol (PCA-20, kmeans init, tied, refine): {current['ari']:.4f}")
    print(f"best variant found: {best['ari']:.4f} "
          f"(n_pcs={best['n_pcs']}, init={best['init']}, cov={best['covariance']}, "
          f"refine={best['refine']})")
    print(f"literature reference: {LITERATURE_ARI:.4f}")
    print(f"remaining unexplained gap at best variant: {LITERATURE_ARI - best['ari']:+.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "sample": SAMPLE, "literature_ari": LITERATURE_ARI,
        "current_protocol": current, "best_variant": best, "all_variants": results,
    }, indent=2))
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
