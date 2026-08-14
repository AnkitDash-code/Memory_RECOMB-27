"""Training loop for the LDCM model."""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.eval.clustering import cluster_embedding as cluster_from_embedding
from src.eval.clustering import consensus_cluster
from src.models.ldcm_memory_layer import LDCMMemoryAutoencoder
from src.models.memory_layer import (
    attention_entropy,
    connectivities_to_edge_index,
    expression_weighted_adjacency,
    usage_entropy,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_positive_neighbors(connectivities):
    edge_index, _ = connectivities_to_edge_index(connectivities)
    rows = edge_index[0].detach().cpu().numpy()
    cols = edge_index[1].detach().cpu().numpy()
    n_nodes = int(max(rows.max(initial=-1), cols.max(initial=-1)) + 1)
    neighbors = [[] for _ in range(n_nodes)]
    for row, col in zip(rows, cols):
        if row != col:
            neighbors[row].append(int(col))
    return neighbors


def _sample_positive_targets(neighbors, seed):
    rng = np.random.default_rng(seed)
    targets = np.empty(len(neighbors), dtype=np.int64)
    for idx, options in enumerate(neighbors):
        if options:
            targets[idx] = int(rng.choice(options))
        else:
            targets[idx] = idx
    return targets


def train_ldcm_model(
    adata,
    epochs=600,
    lr=1e-3,
    weight_decay=0.0,
    memory_slots=16,
    memory_dim=128,
    hidden_dim=256,
    n_hops=4,
    temperature=1.0,
    lambda_usage=0.02,
    lambda_contrastive=0.1,
    expression_weighted=True,
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train the LDCM autoencoder on a preprocessed AnnData."""
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hvg_features = get_hvg_features(adata)
    x = torch.tensor(hvg_features, dtype=torch.float32, device=device)

    if expression_weighted:
        adjacency = expression_weighted_adjacency(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )
    else:
        from src.models.memory_layer import normalized_adjacency

        adjacency = normalized_adjacency(adata.obsp["spatial_connectivities"], device=device)

    neighbors = _build_positive_neighbors(adata.obsp["spatial_connectivities"])

    model = LDCMMemoryAutoencoder(
        feature_dim=x.shape[1],
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        temperature=temperature,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_entropy = math.log(memory_slots)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstruction, embedding, projection, attn_weights = model(x, adjacency)
        recon_loss = F.mse_loss(reconstruction, x)

        projected = F.normalize(projection, dim=-1)
        logits = torch.matmul(projected, projected.T) / temperature
        positive_targets = torch.tensor(
            _sample_positive_targets(neighbors, seed + epoch),
            dtype=torch.long,
            device=device,
        )
        contrastive_loss = F.cross_entropy(logits, positive_targets)

        loss = recon_loss + lambda_contrastive * contrastive_loss - lambda_usage * usage_entropy(attn_weights)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                row = {
                    "epoch": epoch,
                    "recon_loss": recon_loss.item(),
                    "contrastive_loss": contrastive_loss.item(),
                    "total_loss": loss.item(),
                    "median_row_entropy": attention_entropy(attn_weights).median().item(),
                    "usage_entropy": usage_entropy(attn_weights).item(),
                    "max_entropy": max_entropy,
                    "n_slots_used": int(attn_weights.argmax(dim=-1).unique().numel()),
                }
            history.append(row)
            if verbose:
                print(
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"contrastive={row['contrastive_loss']:.4f}  "
                    f"usage_H={row['usage_entropy']:.2f}/{max_entropy:.2f}  "
                    f"slots={row['n_slots_used']}"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, _, _ = model(x, adjacency)

    adata.obsm["X_ldcm"] = embedding.cpu().numpy()
    return model, adata, history


def consensus_from_embeddings(embeddings, n_clusters):
    """Convenience wrapper for label-level consensus across seeds."""
    label_sets = [cluster_from_embedding(embedding, n_clusters) for embedding in embeddings]
    return consensus_cluster(label_sets, n_clusters)