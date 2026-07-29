from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import torch

from src.data.load_visium import load_visium_crop
from src.data.preprocess import preprocess
from src.models.baseline_pca import run_baseline
from src.models.run_graphst import run_graphst
from src.models.train_memory_layer import cluster_from_embedding, train_memory_layer

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_adata = load_visium_crop()
    adata = preprocess(load_visium_crop())
    library_id = next(iter(adata.uns["spatial"].keys()))

    adata = run_baseline(adata)
    n_clusters = adata.obs["leiden"].nunique()

    _, adata, _ = train_memory_layer(adata, epochs=300, device=device)
    adata = cluster_from_embedding(
        adata, "X_memory_trained", "memory_cluster_trained", compute_umap=True
    )

    graphst_adata = run_graphst(raw_adata, n_clusters=n_clusters, device=device)
    # GraphST's own preprocessing subsets to HVGs but keeps the same spots in
    # the same order, so its cluster labels can be copied straight across.
    adata.obs["graphst_cluster"] = graphst_adata.obs["domain"].to_numpy()

    # The crop dataset ships with pre-baked cluster colors keyed to a different
    # category count than what our own clustering produces; drop them so
    # scanpy/squidpy regenerate a palette that actually matches.
    for key in ("leiden_colors", "cluster_colors", "memory_cluster_trained_colors", "graphst_cluster_colors"):
        adata.uns.pop(key, None)

    sq.pl.spatial_scatter(adata, color="memory_cluster_trained", library_id=library_id)
    plt.savefig(FIGURES_DIR / "spatial_memory_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()

    sc.pl.umap(adata, color="memory_cluster_trained", show=False)
    plt.savefig(FIGURES_DIR / "umap_memory_embedding.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sq.pl.spatial_scatter(adata, color="leiden", library_id=library_id, ax=axes[0])
    axes[0].set_title("Baseline (PCA + Leiden)")
    sq.pl.spatial_scatter(adata, color="graphst_cluster", library_id=library_id, ax=axes[1])
    axes[1].set_title("GraphST (Long et al. 2023)")
    sq.pl.spatial_scatter(adata, color="memory_cluster_trained", library_id=library_id, ax=axes[2])
    axes[2].set_title("Trained EmbeddedMemoryLayer + Leiden")
    fig.savefig(FIGURES_DIR / "baseline_vs_memory_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 3 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
