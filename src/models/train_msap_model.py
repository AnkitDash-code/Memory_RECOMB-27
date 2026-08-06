"""Training loop for MaskedSpatialAddressAutoencoder (MSAP).

Key differences from train_spatial_address.py:
  * Loss is ZINB on MASKED spots only, not MSE on all spots.
  * The mask changes every epoch (random 40% by default).
  * Library sizes come from raw counts in adata.layers['counts'].
  * The anti-collapse term (usage_entropy) is kept as a safety net --
    the masking already forces non-trivial propagation, but we've seen
    slot collapse with pure reconstruction objectives before (Stage 2),
    so this defence costs nothing to keep.
  * Inference (mask=None) stores the embedding in adata.obsm['X_msap'].

No existing files are modified. All imported components come from the
existing codebase (count_losses, memory_layer, data.preprocess).
"""

import math
import random

import numpy as np
import scipy.sparse
import torch
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.models.count_losses import zinb_loss
from src.models.memory_layer import (
    attention_entropy,
    expression_weighted_adjacency,
    key_cosine_similarity,
    normalized_adjacency,
    usage_entropy,
)
from src.models.msap_memory_layer import MaskedSpatialAddressAutoencoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_raw_counts(adata):
    """Return the HVG-subset raw count matrix as a dense float32 numpy array.

    preprocess_hvg() stashes raw counts in adata.layers['counts'] before
    normalisation (see src/data/preprocess.py). We extract the HVG columns
    so the count matrix aligns with get_hvg_features()'s output.
    """
    if "counts" not in adata.layers:
        raise ValueError(
            "adata.layers['counts'] not found. "
            "Run preprocess_hvg() before training -- it stashes raw counts "
            "before log-normalisation."
        )
    if "highly_variable" not in adata.var:
        raise ValueError(
            "No 'highly_variable' column in adata.var. "
            "Run preprocess_hvg() first."
        )
    subset = adata[:, adata.var["highly_variable"]]
    x_counts = subset.layers["counts"]
    if scipy.sparse.issparse(x_counts):
        x_counts = x_counts.toarray()
    return np.ascontiguousarray(x_counts, dtype=np.float32)


def train_msap_model(
    adata,
    epochs=600,
    lr=1e-3,
    weight_decay=0.0,
    memory_slots=16,
    memory_dim=128,
    hidden_dim=256,
    max_hops=6,
    temperature=1.0,
    mask_fraction=0.4,
    lambda_usage=0.02,
    expression_weighted=True,
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train MaskedSpatialAddressAutoencoder on a preprocessed (HVG) AnnData.

    Parameters
    ----------
    adata : AnnData
        Must have been processed by preprocess_hvg():
          - adata.layers['counts'] (raw counts before normalisation)
          - adata.var['highly_variable'] (HVG mask)
          - adata.obsp['spatial_connectivities'] (spatial graph)
    mask_fraction : float
        Fraction of spots masked each epoch (default 0.4 = 40%).  Each
        epoch draws a fresh mask so the model cannot memorise which spots
        are always masked.
    lambda_usage : float
        Weight on the marginal usage-entropy anti-collapse term.  Same
        function as in train_spatial_address_model (see Stage 2 notes).
        0.02 is the cross-validated default from Stage 11; keep it here
        rather than re-tuning until the architecture is validated.
    expression_weighted : bool
        If True (default), build the adjacency matrix with expression
        similarity edge weights (Stage 13 fix).  False gives the plain
        row-normalised adjacency -- kept as an ablation option.

    Returns
    -------
    model : MaskedSpatialAddressAutoencoder
    adata : AnnData
        With adata.obsm['X_msap'] set to the inference-time embedding.
    history : list[dict]
        One entry per logged epoch.  Keys match train_spatial_address.py's
        history format for easy comparison.
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Data preparation --
    hvg_features = get_hvg_features(adata)           # (N, n_hvg) log-norm scaled
    x_counts_np = _get_raw_counts(adata)              # (N, n_hvg) raw integer counts
    n_spots, n_genes = hvg_features.shape

    lib_sizes_np = np.asarray(x_counts_np.sum(axis=1), dtype=np.float32).ravel()
    lib_sizes_np = np.maximum(lib_sizes_np, 1.0)      # avoid div-by-zero in decoder

    x = torch.tensor(hvg_features, dtype=torch.float32, device=device)
    x_counts = torch.tensor(x_counts_np, dtype=torch.float32, device=device)
    lib_sizes = torch.tensor(lib_sizes_np, dtype=torch.float32, device=device)

    # -- Spatial graph --
    if expression_weighted:
        adjacency = expression_weighted_adjacency(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )
    else:
        adjacency = normalized_adjacency(
            adata.obsp["spatial_connectivities"], device=device
        )

    # -- Model --
    model = MaskedSpatialAddressAutoencoder(
        feature_dim=n_genes,
        n_genes=n_genes,
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        max_hops=max_hops,
        temperature=temperature,
        zero_inflated=True,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_entropy = math.log(memory_slots)
    history = []

    # -- Training loop --
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Fresh random mask every epoch -- 40% of spots are masked.
        mask = torch.rand(n_spots, device=device) < mask_fraction
        n_masked = int(mask.sum().item())

        # Guard: if by bad luck no spots were masked (very unlikely at 40%),
        # skip the gradient step to avoid a zero-element loss.
        if n_masked == 0:
            continue

        (mu, theta, pi_logits), _, attn_weights = model(x, adjacency, lib_sizes, mask)

        # ZINB loss on masked spots only.
        # theta from CountDecoder is shape (n_genes,) -- a per-gene parameter
        # shared across spots (scVI-style). Expand to (N, n_genes) before
        # indexing so the boolean mask applies correctly to the spot axis.
        theta_expanded = theta.unsqueeze(0).expand(n_spots, -1)  # (N, n_genes)
        recon_loss = zinb_loss(
            x_counts[mask], mu[mask], theta_expanded[mask], pi_logits[mask]
        )

        slot_usage_entropy = usage_entropy(attn_weights)
        # Negative sign: MAXIMISE marginal slot usage (same as baseline).
        loss = recon_loss - lambda_usage * slot_usage_entropy

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                med_entropy = attention_entropy(attn_weights).median().item()
                n_slots_used = int(attn_weights.argmax(dim=-1).unique().numel())
                key_sim = key_cosine_similarity(model.memory.memory_keys).item()
                hop_wts = model.memory.last_hop_weights  # (N, max_hops+1)
                mean_hop = hop_wts.mean(dim=0)           # (max_hops+1,)
                hop_idx = torch.arange(max_hops + 1, device=device, dtype=torch.float32)
                eff_depth = (mean_hop * hop_idx).sum().item()
                hop_entropy = -(mean_hop * torch.log(mean_hop + 1e-12)).sum().item()

            row = {
                "epoch": epoch,
                "n_masked": n_masked,
                "recon_loss": recon_loss.item(),
                "total_loss": loss.item(),
                "median_row_entropy": med_entropy,
                "usage_entropy": slot_usage_entropy.item(),
                "max_entropy": max_entropy,
                "n_slots_used": n_slots_used,
                "key_cosine_similarity": key_sim,
                "effective_hop_depth": eff_depth,
                "hop_weights_mean": mean_hop.tolist(),
                "hop_entropy": hop_entropy,
            }
            log_line = (
                f"epoch {epoch:4d}  zinb={row['recon_loss']:.4f}  "
                f"usage_entropy={row['usage_entropy']:.3f}/{max_entropy:.3f}  "
                f"slots_used={n_slots_used}  "
                f"eff_hop={eff_depth:.2f}  key_cos_sim={key_sim:.3f}"
            )
            history.append(row)
            if verbose:
                print(log_line)

    # -- Inference (no masking) --
    model.eval()
    with torch.no_grad():
        (_, _, _), embedding, attn_weights = model(x, adjacency, lib_sizes, mask=None)

    adata.obsm["X_msap"] = embedding.cpu().numpy()
    return model, adata, history
