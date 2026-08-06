"""Training pipeline for HopfieldMemoryAutoencoder (HMA).

Reuses preprocess and diagnostic functions from src.models.memory_layer
and src.data.preprocess.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.models.hma_memory_layer import HopfieldMemoryAutoencoder
from src.models.memory_layer import (
    attention_entropy,
    expression_weighted_adjacency,
    key_cosine_similarity,
    normalized_adjacency,
    usage_entropy,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_hma_model(
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
    attention_fn="softmax",
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train HopfieldMemoryAutoencoder on a preprocessed (HVG) AnnData.

    Parameters
    ----------
    adata : AnnData
        Preprocessed by preprocess_hvg().
    epochs : int
        Number of training epochs (default 600).
    memory_slots : int
        Number of memory slots M (default 16).
    n_hops : int
        Number of spatial address propagation hops (default 4).
    lambda_usage : float
        Entropy regularization factor to prevent memory slot collapse (default 0.02).
    expression_weighted : bool
        If True, builds expression_weighted_adjacency for graph edges.

    Returns
    -------
    model : HopfieldMemoryAutoencoder
    adata : AnnData
        With adata.obsm['X_hma'] populated.
    history : list[dict]
        Training metric log.
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hvg_features = get_hvg_features(adata)
    x = torch.tensor(hvg_features, dtype=torch.float32, device=device)

    if expression_weighted:
        adjacency = expression_weighted_adjacency(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )
    else:
        adjacency = normalized_adjacency(
            adata.obsp["spatial_connectivities"], device=device
        )

    model = HopfieldMemoryAutoencoder(
        feature_dim=x.shape[1],
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        temperature=temperature,
        attention_fn=attention_fn,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_entropy = math.log(memory_slots)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstruction, _, attn_weights = model(x, adjacency)
        recon_loss = F.mse_loss(reconstruction, x)

        slot_usage_entropy = usage_entropy(attn_weights)
        loss = recon_loss - lambda_usage * slot_usage_entropy

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                median_row_ent = attention_entropy(attn_weights).median().item()
                n_slots_used = int(attn_weights.argmax(dim=-1).unique().numel())
                key_sim = key_cosine_similarity(model.memory.memory_keys).item()
                beta_val = model.memory.attractor_beta.item()

            row = {
                "epoch": epoch,
                "recon_loss": recon_loss.item(),
                "total_loss": loss.item(),
                "median_row_entropy": median_row_ent,
                "usage_entropy": slot_usage_entropy.item(),
                "max_entropy": max_entropy,
                "n_slots_used": n_slots_used,
                "key_cosine_similarity": key_sim,
                "attractor_beta": beta_val,
            }
            history.append(row)

            if verbose:
                log_line = (
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"usage_entropy={row['usage_entropy']:.3f}/{max_entropy:.3f}  "
                    f"slots_used={n_slots_used:2d}  beta={beta_val:.3f}  "
                    f"key_cos_sim={key_sim:.3f}"
                )
                print(log_line)

    model.eval()
    with torch.no_grad():
        _, embedding, _ = model(x, adjacency)

    adata.obsm["X_hma"] = embedding.cpu().numpy()
    return model, adata, history
