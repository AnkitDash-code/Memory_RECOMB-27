import json
import math
from pathlib import Path

import scanpy as sc
import torch
from torch.optim import Adam

from src.models.memory_layer import (
    EmbeddedMemoryAutoencoder,
    attention_entropy,
    connectivities_to_edge_index,
    spatial_smoothness_loss,
)

CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "checkpoints"
LOGS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"


def train_memory_layer(
    adata,
    epochs=300,
    lr=1e-3,
    lambda_spatial=10.0,
    memory_slots=512,
    memory_dim=128,
    device=None,
    log_every=25,
):
    """lambda_spatial=10.0 and epochs=300 were chosen via a sweep on the crop
    dataset (lambda in [0.1, 20], epochs in [300, 1000]): lambda=10 gave the
    best spatial-coherence/silhouette tradeoff, and training past ~300 epochs
    kept improving reconstruction loss while *degrading* both clustering
    metrics (overfitting reconstruction at the expense of structure)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.data.preprocess import get_pca_features

    x = torch.tensor(get_pca_features(adata), dtype=torch.float32).to(device)
    edge_index, edge_weight = connectivities_to_edge_index(adata.obsp["spatial_connectivities"])
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)

    model = EmbeddedMemoryAutoencoder(
        feature_dim=x.shape[1], memory_slots=memory_slots, memory_dim=memory_dim
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    max_entropy = math.log(memory_slots)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstruction, embedding, attn_weights = model(x)
        recon_loss = torch.nn.functional.mse_loss(reconstruction, x)
        spatial_loss = spatial_smoothness_loss(embedding, edge_index, edge_weight)
        loss = recon_loss + lambda_spatial * spatial_loss

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                median_entropy = attention_entropy(attn_weights).median().item()
            row = {
                "epoch": epoch,
                "recon_loss": recon_loss.item(),
                "spatial_loss": spatial_loss.item(),
                "total_loss": loss.item(),
                "median_entropy": median_entropy,
                "max_entropy": max_entropy,
            }
            history.append(row)
            print(
                f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                f"spatial={row['spatial_loss']:.4f}  total={row['total_loss']:.4f}  "
                f"median_entropy={median_entropy:.4f}/{max_entropy:.4f}"
            )

    model.eval()
    with torch.no_grad():
        _, embedding, attn_weights = model(x)
    adata.obsm["X_memory_trained"] = embedding.cpu().numpy()

    final_entropy = attention_entropy(attn_weights).median().item()
    if final_entropy < 0.05 * max_entropy:
        print(f"WARNING: final median entropy {final_entropy:.4f} near zero -> slot collapse")

    return model, adata, history


def cluster_from_embedding(adata, obsm_key, cluster_key, compute_umap=False, n_clusters_target=None):
    from src.eval.metrics import search_leiden_resolution

    neighbors_key = f"{cluster_key}_neighbors"
    sc.pp.neighbors(adata, use_rep=obsm_key, key_added=neighbors_key)

    resolution = 1.0
    if n_clusters_target is not None:
        resolution = search_leiden_resolution(adata, neighbors_key, n_clusters_target)

    sc.tl.leiden(adata, neighbors_key=neighbors_key, key_added=cluster_key, resolution=resolution)
    if compute_umap:
        sc.tl.umap(adata, neighbors_key=neighbors_key)
    return adata


def main():
    from src.data.load_visium import load_visium_crop, load_visium_full
    from src.data.preprocess import preprocess

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    for name, loader in [("crop", load_visium_crop), ("full", load_visium_full)]:
        print(f"\n=== training on {name} ===")
        adata = preprocess(loader())
        model, adata, history = train_memory_layer(adata)

        adata = cluster_from_embedding(adata, "X_memory_trained", "memory_cluster_trained")
        n_clusters = adata.obs["memory_cluster_trained"].nunique()
        print(f"{name}: {n_clusters} clusters from trained embedding")

        torch.save(model.state_dict(), CHECKPOINTS_DIR / f"memory_layer_{name}.pt")
        (LOGS_DIR / f"train_history_{name}.json").write_text(json.dumps(history, indent=2))
        adata.write_h5ad(CHECKPOINTS_DIR / f"adata_trained_{name}.h5ad")


if __name__ == "__main__":
    main()
