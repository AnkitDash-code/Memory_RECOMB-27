"""Training loop for AGAPMemoryAutoencoder."""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.eval.clustering import cluster_embedding as cluster_from_embedding
from src.eval.clustering import consensus_cluster
from src.models.agap_memory_layer import AGAPMemoryAutoencoder
from src.models.memory_layer import (
    attention_entropy,
    connectivities_to_edge_index,
    usage_entropy,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_agap_model(
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
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train AGAP on a preprocessed AnnData object."""
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hvg_features = get_hvg_features(adata)
    x = torch.tensor(hvg_features, dtype=torch.float32, device=device)
    edge_index, _ = connectivities_to_edge_index(adata.obsp["spatial_connectivities"])
    edge_index = edge_index.to(device)

    model = AGAPMemoryAutoencoder(
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

        reconstruction, embedding, attn_weights = model(x, edge_index)
        recon_loss = F.mse_loss(reconstruction, x)
        attn_usage = usage_entropy(attn_weights)
        loss = recon_loss - lambda_usage * attn_usage

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
                "usage_entropy": attn_usage.item(),
                "max_entropy": max_entropy,
                "n_slots_used": n_slots_used,
            }
            history.append(row)

            if verbose:
                print(
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"usage_entropy={row['usage_entropy']:.3f}/{max_entropy:.3f}  "
                    f"slots_used={n_slots_used:2d}"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, _ = model(x, edge_index)

    adata.obsm["X_agap"] = embedding.cpu().numpy()
    return model, adata, history


def consensus_from_embeddings(embeddings, n_clusters):
    """Convenience wrapper for label-level consensus across seeds."""
    label_sets = [cluster_from_embedding(embedding, n_clusters) for embedding in embeddings]
    return consensus_cluster(label_sets, n_clusters)