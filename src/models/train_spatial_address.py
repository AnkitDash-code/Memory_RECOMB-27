"""Training loop for SpatialAddressMemoryAutoencoder.

Separate from train_memory_layer.py, which trains the original Phase 0
PCA-input model. That model is kept intact as an ablation baseline; this one
consumes real HVG expression and propagates addresses over the spatial graph.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.models.memory_layer import (
    SpatialAddressMemoryAutoencoder,
    attention_entropy,
    normalized_adjacency,
    usage_entropy,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_spatial_address_model(
    adata,
    epochs=600,
    lr=1e-3,
    weight_decay=0.0,
    memory_slots=32,
    memory_dim=128,
    hidden_dim=256,
    n_hops=4,
    temperature=1.0,
    feature_hops=0,
    latent_hops=0,
    lambda_usage=0.1,
    lambda_sharpen=0.0,
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train the address-propagation model on a preprocessed (HVG) AnnData.

    lambda_usage > 0 maximizes the entropy of MARGINAL slot usage, which is what
    stops slot collapse. With it at 0 the model reliably collapses to a single
    slot decoding the dataset mean (measured: slots_used=1, ARI=0.0), so it
    defaults on rather than off.

    lambda_sharpen > 0 additionally *minimizes* per-row entropy, pushing each
    individual spot to commit to a slot. The two terms pull in complementary
    directions: spread usage across the codebook, but keep each spot's own
    assignment confident.

    Defaults are the tuned configuration from the DLPFC 151673 sweeps
    (0.5713 ± 0.0057 over 5 seeds):

      memory_slots=32  -- a compact codebook clearly beats a large one when there
                          are only ~7 true domains. Measured: 32 -> 0.569,
                          64 -> 0.542, 128 -> 0.515, 256 -> 0.408, and going
                          below 32 (8/16) became unstable across seeds.
      n_hops=4         -- ARI rises monotonically with address-propagation hops
                          (1 -> 0.489, 2 -> 0.538, 4 -> 0.549).
      lambda_usage=0.1 -- enough to prevent collapse; larger values
                          over-regularize toward uniform usage (1.0 -> ~0.34,
                          5.0 -> ~0.22).

    feature_hops/latent_hops default to 0, the pure formulation. Both hybrid
    variants were tested and lost to it -- see outputs/logs/stage2_progress.md.
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(get_hvg_features(adata), dtype=torch.float32).to(device)
    adjacency = normalized_adjacency(adata.obsp["spatial_connectivities"], device=device)

    model = SpatialAddressMemoryAutoencoder(
        feature_dim=x.shape[1],
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        temperature=temperature,
        feature_hops=feature_hops,
        latent_hops=latent_hops,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_entropy = math.log(memory_slots)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstruction, _, attn_weights = model(x, adjacency)
        recon_loss = F.mse_loss(reconstruction, x)

        loss = recon_loss
        slot_usage_entropy = usage_entropy(attn_weights)
        if lambda_usage:
            # Negative sign: MAXIMIZE marginal usage entropy -> use all slots.
            loss = loss - lambda_usage * slot_usage_entropy
        if lambda_sharpen:
            # Positive sign: MINIMIZE per-row entropy -> confident per-spot assignment.
            loss = loss + lambda_sharpen * attention_entropy(attn_weights).mean()

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                median_entropy = attention_entropy(attn_weights).median().item()
                n_slots_used = (attn_weights.argmax(dim=-1).unique()).numel()
            row = {
                "epoch": epoch,
                "recon_loss": recon_loss.item(),
                "total_loss": loss.item(),
                "median_entropy": median_entropy,
                "usage_entropy": slot_usage_entropy.item(),
                "max_entropy": max_entropy,
                "n_slots_used": int(n_slots_used),
            }
            history.append(row)
            if verbose:
                print(
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"row_entropy={median_entropy:.3f}  "
                    f"usage_entropy={row['usage_entropy']:.3f}/{max_entropy:.3f}  "
                    f"slots_used={n_slots_used}"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, attn_weights = model(x, adjacency)

    adata.obsm["X_spatial_address"] = embedding.cpu().numpy()
    return model, adata, history
