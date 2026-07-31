"""Visualize ground truth vs. baseline vs. GraphST vs. our tuned model on DLPFC 151673.

Supersedes src/viz/spatial_plots.py's story: that script visualizes the Phase 0
model (PCA input, no address propagation) on the label-free Visium crop dataset.
This one uses the current, fully cross-validated architecture
(SpatialAddressMemoryAutoencoder, train_spatial_address_model's defaults --
memory_slots=16, n_hops=4, lambda_usage=0.02, expression_weighted=True) on the
dataset that actually has ground truth, and puts every method through the same clustering protocol
(src/eval/clustering.py) so the panels are a fair visual comparison, not just
four different pipelines. Single seed (0), single slice -- see the headline
12-slice/5-seed numbers in README.md for the real result this illustrates.
"""

from pathlib import Path

import matplotlib
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from src.data.load_dlpfc import load_dlpfc_151673
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.models.baseline_pca import run_baseline
from src.models.run_graphst import run_graphst
from src.models.train_spatial_address import train_spatial_address_model

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_dlpfc_151673()
    adata = preprocess_hvg(load_dlpfc_151673())
    n_layers = adata.obs["ground_truth_layer"].nunique()
    coords = adata.obsm["spatial"]
    library_id = next(iter(adata.uns["spatial"].keys()))

    # Baseline: PCA + our shared clustering protocol (not the old Leiden path).
    baseline_adata = run_baseline(adata.copy(), n_clusters_target=n_layers)
    adata.obs["baseline_cluster"] = cluster_embedding(
        baseline_adata.obsm["X_pca"], n_layers, coords=coords, refine=True
    )

    # GraphST, same protocol.
    graphst_adata = run_graphst(raw.copy(), n_clusters=n_layers, device=device, cluster=False)
    adata.obs["graphst_cluster"] = cluster_embedding(
        graphst_adata.obsm["emb"], n_layers, coords=graphst_adata.obsm["spatial"], refine=True
    )

    # Ours: tuned SpatialAddressMemoryAutoencoder (memory_slots=32 default).
    _, trained, _ = train_spatial_address_model(adata.copy(), seed=0, device=device)
    adata.obs["ours_cluster"] = cluster_embedding(
        trained.obsm["X_spatial_address"], n_layers, coords=coords, refine=True
    )

    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    for column in ("baseline_cluster", "graphst_cluster", "ours_cluster"):
        ari = adjusted_rand_score(truth[mask], adata.obs[column].to_numpy()[mask])
        print(f"{column}: ARI = {ari:.4f}")

    for column in ("ground_truth_layer", "baseline_cluster", "graphst_cluster", "ours_cluster"):
        adata.obs[column] = adata.obs[column].astype("category")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    panels = [
        ("ground_truth_layer", "Ground truth (Maynard et al. 2021)"),
        ("baseline_cluster", "Baseline (PCA + mclust-equiv.)"),
        ("graphst_cluster", "GraphST (Long et al. 2023)"),
        ("ours_cluster", "Ours (SpatialAddressMemoryAutoencoder)"),
    ]
    for ax, (column, title) in zip(axes, panels):
        import squidpy as sq

        sq.pl.spatial_scatter(adata, color=column, library_id=library_id, ax=ax)
        ax.set_title(title)
    fig.savefig(FIGURES_DIR / "dlpfc_ground_truth_vs_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved outputs/figures/dlpfc_ground_truth_vs_methods.png")


if __name__ == "__main__":
    main()
