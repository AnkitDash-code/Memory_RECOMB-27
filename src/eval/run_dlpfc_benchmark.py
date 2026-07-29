import json
import time
from pathlib import Path

import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_151673
from src.data.preprocess import preprocess
from src.models.baseline_pca import run_baseline
from src.models.run_graphst import run_graphst
from src.models.train_memory_layer import cluster_from_embedding, train_memory_layer

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "dlpfc_ari_results.json"

# Real ARI scores computed directly (not cited) from the ground-truth-labeled
# predictions released with Kang et al. 2025 (Nucleic Acids Research) --
# see scratch/compute_real_ari.py for the exact computation. Included here
# for a side-by-side, apples-to-apples comparison on the SAME dataset/labels.
LITERATURE_ARI_151673 = {
    "GraphST": 0.6327,
    "STAGATE": 0.5892,
    "Spatial_MGCN": 0.5561,
    "BayesSpace": 0.5499,
    "DeepST": 0.5384,
    "conST": 0.5277,
    "STMGCN": 0.5107,
    "SEDR": 0.4723,
    "SpaGCN": 0.4652,
    "Seurat": 0.4295,
    "stLearn": 0.3681,
    "CCST": 0.3563,
    "SpaceFlow": 0.3510,
    "SCGDL": 0.3216,
}


def ari_vs_ground_truth(adata, cluster_key):
    mask = adata.obs["ground_truth_layer"].notna()
    return adjusted_rand_score(
        adata.obs["ground_truth_layer"][mask], adata.obs[cluster_key][mask]
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_adata = load_dlpfc_151673()
    adata = preprocess(load_dlpfc_151673())
    n_ground_truth_layers = adata.obs["ground_truth_layer"].nunique()
    print(f"n_obs after preprocessing: {adata.n_obs}, ground-truth layers: {n_ground_truth_layers}")

    results = {}

    start = time.time()
    adata = run_baseline(adata, n_clusters_target=n_ground_truth_layers)
    results["Scanpy PCA+Leiden (baseline)"] = {
        "ari_vs_ground_truth": ari_vs_ground_truth(adata, "leiden"),
        "n_clusters": int(adata.obs["leiden"].nunique()),
        "wall_time_s": round(time.time() - start, 3),
    }

    start = time.time()
    _, adata, history = train_memory_layer(adata, epochs=300, device=device)
    adata = cluster_from_embedding(
        adata, "X_memory_trained", "memory_cluster_trained", n_clusters_target=n_ground_truth_layers
    )
    results["EmbeddedMemoryLayer (trained)"] = {
        "ari_vs_ground_truth": ari_vs_ground_truth(adata, "memory_cluster_trained"),
        "n_clusters": int(adata.obs["memory_cluster_trained"].nunique()),
        "wall_time_s": round(time.time() - start, 3),
        "final_recon_loss": history[-1]["recon_loss"],
        "final_median_entropy": history[-1]["median_entropy"],
    }

    start = time.time()
    graphst_adata = run_graphst(raw_adata, n_clusters=n_ground_truth_layers, device=device)
    graphst_adata.obs["ground_truth_layer"] = adata.obs["ground_truth_layer"].reindex(
        graphst_adata.obs_names
    )
    results["GraphST (local run)"] = {
        "ari_vs_ground_truth": ari_vs_ground_truth(graphst_adata, "domain"),
        "n_clusters": int(graphst_adata.obs["domain"].nunique()),
        "wall_time_s": round(time.time() - start, 3),
    }

    print("\n=== Our methods, real ARI vs. spatialLIBD ground truth (DLPFC 151673) ===")
    for method, row in results.items():
        print(f"  {method:40s} ARI = {row['ari_vs_ground_truth']:.4f}  n_clusters={row['n_clusters']}")

    print("\n=== Literature (Kang et al. 2025), same dataset/labels, computed directly from their released predictions ===")
    for method, ari in sorted(LITERATURE_ARI_151673.items(), key=lambda kv: -kv[1]):
        print(f"  {method:40s} ARI = {ari:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "our_methods": results,
        "literature_kang_et_al_2025": LITERATURE_ARI_151673,
    }, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
