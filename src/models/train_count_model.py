"""Stage 3 training loop: address propagation + NB/ZINB count likelihood + contrastive regularization."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.models.count_losses import nb_loss, zinb_loss
from src.models.memory_layer import (
    SpatialAddressCountAutoencoder,
    attention_entropy,
    normalized_adjacency,
    usage_entropy,
)
from src.models.train_spatial_address import set_seed


def _hvg_counts(adata):
    """Raw counts restricted to the highly-variable genes, as a dense array."""
    subset = adata[:, adata.var["highly_variable"]]
    counts = subset.layers["counts"]
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    return np.ascontiguousarray(counts, dtype=np.float32)


def contrastive_address_loss(attn_weights, attn_corrupted):
    """Penalize agreement between real and feature-corrupted address assignments.

    GraphST and MAEST both report that a contrastive/denoising term is needed to
    stop the representation collapsing. Here the discrimination is done in
    address space: a spot's address under real features should NOT match its
    address when the features are shuffled across spots. Implemented as the
    mean dot-product similarity between the two distributions, which is
    minimized.
    """
    return (attn_weights * attn_corrupted).sum(dim=-1).mean()


def train_count_model(
    adata,
    epochs=600,
    lr=1e-3,
    memory_slots=64,
    memory_dim=128,
    hidden_dim=256,
    n_hops=4,
    lambda_usage=0.1,
    lambda_contrastive=0.0,
    zero_inflated=True,
    seed=0,
    device=None,
    log_every=150,
    verbose=True,
):
    """Train the count-likelihood variant. Returns (model, adata, history).

    Input features stay the scaled HVG matrix (a well-conditioned encoder input);
    the *reconstruction target* is raw HVG counts under an NB/ZINB likelihood.
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(get_hvg_features(adata), dtype=torch.float32).to(device)
    counts = torch.tensor(_hvg_counts(adata), dtype=torch.float32).to(device)
    library_size = counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    adjacency = normalized_adjacency(adata.obsp["spatial_connectivities"], device=device)

    model = SpatialAddressCountAutoencoder(
        feature_dim=x.shape[1],
        n_genes=counts.shape[1],
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        zero_inflated=zero_inflated,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    max_entropy = math.log(memory_slots)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        (mu, theta, pi_logits), _, attn_weights = model(x, library_size, adjacency)
        recon_loss = (
            zinb_loss(counts, mu, theta, pi_logits)
            if zero_inflated
            else nb_loss(counts, mu, theta)
        )

        loss = recon_loss - lambda_usage * usage_entropy(attn_weights)

        if lambda_contrastive:
            permutation = torch.randperm(x.shape[0], device=device)
            _, _, attn_corrupted = model(x[permutation], library_size, adjacency)
            loss = loss + lambda_contrastive * contrastive_address_loss(
                attn_weights, attn_corrupted
            )

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                row = {
                    "epoch": epoch,
                    "recon_loss": recon_loss.item(),
                    "total_loss": loss.item(),
                    "median_entropy": attention_entropy(attn_weights).median().item(),
                    "usage_entropy": usage_entropy(attn_weights).item(),
                    "max_entropy": max_entropy,
                    "n_slots_used": int(attn_weights.argmax(dim=-1).unique().numel()),
                }
            history.append(row)
            if verbose:
                print(
                    f"epoch {epoch:4d}  nll={row['recon_loss']:.4f}  "
                    f"usage_H={row['usage_entropy']:.2f}/{max_entropy:.2f}  "
                    f"slots={row['n_slots_used']}"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, _ = model(x, library_size, adjacency)
    adata.obsm["X_count_address"] = embedding.cpu().numpy()
    return model, adata, history
