"""Training loop for the physical-scale Hop-Fusion ablation.

This module is intentionally separate from ``train_spatial_address.py``.  The
existing fixed-hop model remains the current comparator, while this trainer
implements the new concat-fusion mechanism and the optional, isolated
address-coherence loss.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.physical_scale import (
    get_average_edge_length_um,
    local_expression_heterogeneity,
    um_radius_to_hop_count,
)
from src.data.preprocess import get_hvg_features
from src.models.memory_layer import (
    HopFusionMemoryAutoencoder,
    address_spatial_coherence_loss,
    attention_entropy,
    expression_similarity_edge_weights,
    expression_weighted_adjacency,
    normalized_adjacency,
    usage_entropy,
)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_train_mask(train_mask, n_obs):
    if train_mask is None:
        return np.ones(n_obs, dtype=bool)
    mask = np.asarray(train_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != n_obs:
        raise ValueError("train_mask must be a boolean vector with one value per observation")
    if not mask.any():
        raise ValueError("train_mask must contain at least one training observation")
    return mask


def mean_address_entropy_by_hop(addresses_by_hop, mask=None):
    """Return the mean per-spot address entropy at every propagation depth.

    This deliberately measures row-wise entropy rather than marginal slot
    usage.  The former tells us whether each spot makes a discriminative memory
    assignment; the latter only tells us whether the codebook is used evenly
    across the dataset.
    """
    if not addresses_by_hop:
        raise ValueError("addresses_by_hop must contain at least one hop view")
    return torch.stack([
        attention_entropy(addresses if mask is None else addresses[mask]).mean()
        for addresses in addresses_by_hop
    ])


def train_hop_fusion_model(
    adata,
    *,
    platform="visium",
    physical_radius_um=None,
    max_hops=None,
    memory_slots=16,
    memory_dim=128,
    hidden_dim=256,
    fusion_hidden_dim=128,
    fusion_depth=2,
    temperature=1.0,
    attention_fn="softmax",
    epochs=600,
    lr=1e-3,
    weight_decay=0.0,
    lambda_usage=0.02,
    lambda_sharpen=0.0,
    lambda_spatial_coherence=0.0,
    expression_weighted=True,
    train_mask=None,
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train Hop-Fusion on an HVG-preprocessed AnnData object.

    ``physical_radius_um`` is the portable hyperparameter.  ``max_hops`` is
    accepted only as a convenience for the DLPFC selection stage: when no
    radius is supplied, it is converted to a radius using that dataset's
    measured edge length.  In both cases the runtime fusion window is always
    derived by :func:`um_radius_to_hop_count`.

    ``train_mask`` supports the breast-cancer contiguous-block holdout.  The
    graph is still evaluated over all observations (a transductive spatial
    embedding), but reconstruction, usage, and optional coherence losses are
    computed only on the training block.

    ``lambda_sharpen`` is an opt-in diagnostic regularizer.  It minimizes the
    average per-spot address entropy across *all* hop views, restoring an
    explicit incentive for confident addresses when the concat-fusion decoder
    can otherwise reconstruct from a wide mixture of diffuse views.
    """
    _set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hvg_features = get_hvg_features(adata)
    n_obs = hvg_features.shape[0]
    train_mask_np = _as_train_mask(train_mask, n_obs)
    train_mask_t = torch.tensor(train_mask_np, dtype=torch.bool, device=device)

    avg_edge_length_um = get_average_edge_length_um(adata, platform)
    if physical_radius_um is None:
        if max_hops is None:
            raise ValueError("provide physical_radius_um or the DLPFC reference max_hops")
        physical_radius_um = float(max_hops) * avg_edge_length_um
    physical_radius_um = float(physical_radius_um)
    fusion_hops = um_radius_to_hop_count(
        physical_radius_um, platform, avg_edge_length_um
    )
    heterogeneity, heterogeneity_hops = local_expression_heterogeneity(
        hvg_features,
        adata.obsp["spatial_connectivities"],
        physical_radius_um,
        platform,
        avg_edge_length_um,
    )

    x = torch.tensor(hvg_features, dtype=torch.float32, device=device)
    score = torch.tensor(heterogeneity, dtype=torch.float32, device=device)
    if expression_weighted:
        adjacency = expression_weighted_adjacency(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )
    else:
        adjacency = normalized_adjacency(adata.obsp["spatial_connectivities"], device=device)

    coherence_edge_index = coherence_edge_weight = None
    if lambda_spatial_coherence:
        coherence_edge_index, coherence_edge_weight = expression_similarity_edge_weights(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )
        if coherence_edge_index.numel():
            keep = train_mask_np[
                coherence_edge_index[0].cpu().numpy()
            ] & train_mask_np[coherence_edge_index[1].cpu().numpy()]
            keep = torch.tensor(keep, dtype=torch.bool, device=device)
            coherence_edge_index = coherence_edge_index[:, keep]
            coherence_edge_weight = coherence_edge_weight[keep]

    model = HopFusionMemoryAutoencoder(
        feature_dim=x.shape[1],
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        max_hops=fusion_hops,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_depth=fusion_depth,
        temperature=temperature,
        attention_fn=attention_fn,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    max_entropy = math.log(memory_slots)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstruction, _, addresses = model(x, adjacency, score)
        recon_loss = F.mse_loss(reconstruction[train_mask_t], x[train_mask_t])
        slot_usage_entropy = usage_entropy(addresses[train_mask_t])
        loss = recon_loss - lambda_usage * slot_usage_entropy
        sharpen_loss = mean_address_entropy_by_hop(
            model.memory.last_address_by_hop, train_mask_t
        ).mean()
        if lambda_sharpen:
            loss = loss + lambda_sharpen * sharpen_loss
        coherence_loss = None
        if lambda_spatial_coherence:
            coherence_loss = address_spatial_coherence_loss(
                addresses, coherence_edge_index, coherence_edge_weight
            )
            loss = loss + lambda_spatial_coherence * coherence_loss

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                row_entropy = attention_entropy(addresses[train_mask_t]).median().item()
                mean_entropy_by_hop = mean_address_entropy_by_hop(
                    model.memory.last_address_by_hop, train_mask_t
                )
                median_entropy_by_hop = torch.stack([
                    attention_entropy(view[train_mask_t]).median()
                    for view in model.memory.last_address_by_hop
                ])
                n_slots_used = int(addresses[train_mask_t].argmax(dim=-1).unique().numel())
            row = {
                "epoch": epoch,
                "recon_loss": float(recon_loss.item()),
                "total_loss": float(loss.item()),
                "median_entropy": float(row_entropy),
                "usage_entropy": float(slot_usage_entropy.item()),
                "max_entropy": float(max_entropy),
                "n_slots_used": n_slots_used,
                "mean_entropy_by_hop": mean_entropy_by_hop.detach().cpu().tolist(),
                "median_entropy_by_hop": median_entropy_by_hop.detach().cpu().tolist(),
                "physical_radius_um": physical_radius_um,
                "average_edge_length_um": float(avg_edge_length_um),
                "fusion_hops": int(fusion_hops),
                "heterogeneity_hops": int(heterogeneity_hops),
            }
            if lambda_sharpen:
                row["sharpen_loss"] = float(sharpen_loss.item())
            if coherence_loss is not None:
                row["spatial_coherence_loss"] = float(coherence_loss.item())
            history.append(row)
            if verbose:
                message = (
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"fusion_hops={fusion_hops}  heterogeneity_hops={heterogeneity_hops}  "
                    f"slots_used={n_slots_used}"
                )
                if coherence_loss is not None:
                    message += f"  coherence={row['spatial_coherence_loss']:.4f}"
                print(message)

    model.eval()
    with torch.no_grad():
        _, embedding, addresses = model(x, adjacency, score)
    embedding_np = embedding.cpu().numpy()
    adata.obsm["X_hop_fusion"] = embedding_np
    # This alias makes downstream clustering helpers interchangeable between
    # the current fixed-hop model and the new architecture.
    adata.obsm["X_spatial_address"] = embedding_np
    adata.uns["hop_fusion"] = {
        "platform": platform,
        "physical_radius_um": physical_radius_um,
        "average_edge_length_um": float(avg_edge_length_um),
        "fusion_hops": int(fusion_hops),
        "heterogeneity_hops": int(heterogeneity_hops),
        "fusion_hidden_dim": int(fusion_hidden_dim),
        "fusion_depth": int(fusion_depth),
        "train_fraction": float(train_mask_np.mean()),
    }
    return model, adata, history
