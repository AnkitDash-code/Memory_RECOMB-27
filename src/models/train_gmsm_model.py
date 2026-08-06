"""Training loop for the Gated Multi-Scale Memory (GMSM) model."""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.eval.clustering import cluster_embedding as cluster_from_embedding
from src.eval.clustering import consensus_cluster
from src.models.gmsm_memory_layer import GatedMultiScaleMemoryAutoencoder
from src.models.memory_layer import (
    attention_entropy,
    expression_weighted_adjacency,
    normalized_adjacency,
    usage_entropy,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_gmsm_model(
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
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train the GMSM autoencoder on a preprocessed AnnData."""
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hvg_features = get_hvg_features(adata)
    x = torch.tensor(hvg_features, dtype=torch.float32, device=device)

    if expression_weighted:
        adjacency = expression_weighted_adjacency(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )
    else:
        adjacency = normalized_adjacency(adata.obsp["spatial_connectivities"], device=device)

    model = GatedMultiScaleMemoryAutoencoder(
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

        reconstruction, embedding, a_local, a_global, gate = model(x, adjacency)
        recon_loss = F.mse_loss(reconstruction, x)
        local_usage = usage_entropy(a_local)
        global_usage = usage_entropy(a_global)
        loss = recon_loss - lambda_usage * local_usage - lambda_usage * global_usage

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                local_row_entropy = attention_entropy(a_local).median().item()
                global_row_entropy = attention_entropy(a_global).median().item()
                local_slots_used = int(a_local.argmax(dim=-1).unique().numel())
                global_slots_used = int(a_global.argmax(dim=-1).unique().numel())
                mean_gate = gate.mean().item()

            row = {
                "epoch": epoch,
                "recon_loss": recon_loss.item(),
                "total_loss": loss.item(),
                "local_row_entropy": local_row_entropy,
                "global_row_entropy": global_row_entropy,
                "local_usage_entropy": local_usage.item(),
                "global_usage_entropy": global_usage.item(),
                "max_entropy": max_entropy,
                "local_slots_used": local_slots_used,
                "global_slots_used": global_slots_used,
                "mean_gate": mean_gate,
            }
            history.append(row)

            if verbose:
                print(
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"local_usage={row['local_usage_entropy']:.3f}/{max_entropy:.3f}  "
                    f"global_usage={row['global_usage_entropy']:.3f}/{max_entropy:.3f}  "
                    f"gate={mean_gate:.3f}  "
                    f"slots_local={local_slots_used:2d}  slots_global={global_slots_used:2d}"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, _, _, gate = model(x, adjacency)

    adata.obsm["X_gmsm"] = embedding.cpu().numpy()
    return model, adata, history


def consensus_from_embeddings(embeddings, n_clusters):
    """Convenience wrapper for label-level consensus across seeds."""
    label_sets = [cluster_from_embedding(embedding, n_clusters) for embedding in embeddings]
    return consensus_cluster(label_sets, n_clusters)