"""Training loop for Zero-Inflated Spatial Memory (ZISM).

This isolates the ZINB loss on top of the stable expression-weighted,
fixed-hop address-propagation baseline.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.eval.clustering import cluster_embedding as cluster_from_embedding
from src.eval.clustering import consensus_cluster
from src.models.count_losses import zinb_loss
from src.models.memory_layer import (
    SpatialAddressCountAutoencoder,
    attention_entropy,
    expression_weighted_adjacency,
    usage_entropy,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _count_matrix(adata):
    """Return the raw count matrix as a contiguous float32 array.

    Prefer adata.raw.X if present, then the counts layer created by
    preprocess_hvg(), and only fall back to adata.X as a last resort.
    """
    if getattr(adata, "raw", None) is not None:
        counts = adata.raw.X
    elif "counts" in adata.layers:
        counts = adata.layers["counts"]
    else:
        counts = adata.X
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    return np.ascontiguousarray(counts, dtype=np.float32)


def train_zism_model(
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
    expression_weighted=True,
    zero_inflated=True,
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train the zero-inflated spatial memory model on a preprocessed AnnData."""
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hvg_features = get_hvg_features(adata)
    x = torch.tensor(hvg_features, dtype=torch.float32, device=device)

    count_matrix = _count_matrix(adata)
    x_counts = torch.tensor(count_matrix, dtype=torch.float32, device=device)
    lib_sizes = x_counts.sum(dim=1).clamp(min=1.0).unsqueeze(-1)

    if expression_weighted:
        adjacency = expression_weighted_adjacency(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )
    else:
        from src.models.memory_layer import normalized_adjacency

        adjacency = normalized_adjacency(adata.obsp["spatial_connectivities"], device=device)

    model = SpatialAddressCountAutoencoder(
        feature_dim=x.shape[1],
        n_genes=x_counts.shape[1],
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        temperature=temperature,
        zero_inflated=zero_inflated,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_entropy = math.log(memory_slots)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        (mu, theta, pi_logits), embedding, attn_weights = model(x, lib_sizes, adjacency)
        recon_loss = zinb_loss(x_counts, mu, theta, pi_logits)
        loss = recon_loss - lambda_usage * usage_entropy(attn_weights)

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                median_row_entropy = attention_entropy(attn_weights).median().item()
                n_slots_used = int(attn_weights.argmax(dim=-1).unique().numel())

            row = {
                "epoch": epoch,
                "recon_loss": recon_loss.item(),
                "total_loss": loss.item(),
                "median_row_entropy": median_row_entropy,
                "usage_entropy": usage_entropy(attn_weights).item(),
                "max_entropy": max_entropy,
                "n_slots_used": n_slots_used,
            }
            history.append(row)

            if verbose:
                print(
                    f"epoch {epoch:4d}  zinb_nll={row['recon_loss']:.4f}  "
                    f"usage_H={row['usage_entropy']:.2f}/{max_entropy:.2f}  "
                    f"slots={row['n_slots_used']}"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, _ = model(x, lib_sizes, adjacency)

    adata.obsm["X_zism"] = embedding.cpu().numpy()
    return model, adata, history


def consensus_from_embeddings(embeddings, n_clusters):
    """Convenience wrapper for label-level consensus across seeds."""
    label_sets = [cluster_from_embedding(embedding, n_clusters) for embedding in embeddings]
    return consensus_cluster(label_sets, n_clusters)