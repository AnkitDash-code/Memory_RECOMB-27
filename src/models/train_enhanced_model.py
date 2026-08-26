"""Training loop for the enhanced two-stream memory architecture.

Keeps the existing train_spatial_address_model ENTIRELY INTACT as the
baseline.  This function is a new entry-point that wires up the 5 fixes
from enhanced_memory_layer.py as explicit, independently-toggled terms
so each can be ablated cleanly.

Default config (all fixes ON except dynamic adjacency):
  lambda_repulsion     = 0.05   Fix 1 -- key repulsion
  lambda_kl_contrastive= 0.01   Fix 2 -- KL contrastive views
  adj_refresh_every    = 0      Fix 3 -- off by default (expensive, test separately)
  use_two_stream       = True   Fix 4 -- domain + state banks
  entropy_gate         = True   Fix 5 -- certainty-scaled propagation depth
  n_domain_slots       = 8      } total = 16, matches cross-validated count
  n_state_slots        = 8      }

To reproduce the original model exactly set all lambdas to 0 and
use_two_stream=False, entropy_gate=False.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.models.enhanced_memory_layer import (
    TwoStreamMemoryAutoencoder,
    augment_adjacency,
    augment_features,
    key_repulsion_loss,
    kl_contrastive_address_loss,
)
from src.models.memory_layer import (
    SpatialAddressMemoryAutoencoder,
    attention_entropy,
    expression_weighted_adjacency,
    key_cosine_similarity,
    normalized_adjacency,
    usage_entropy,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_enhanced_model(
    adata,
    # ---------- training basics ----------
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    memory_dim: int = 128,
    hidden_dim: int = 256,
    n_hops: int = 4,
    temperature: float = 1.0,
    lambda_usage: float = 0.02,
    expression_weighted: bool = True,
    seed: int = 0,
    device=None,
    log_every: int = 100,
    verbose: bool = True,
    # ---------- Fix 1: Key Repulsion ----------
    lambda_repulsion: float = 0.05,
    # ---------- Fix 2: KL Contrastive ----------
    lambda_kl_contrastive: float = 0.01,
    adj_drop_rate: float = 0.10,
    feat_mask_rate: float = 0.05,
    # ---------- Fix 3: Dynamic adjacency ----------
    adj_refresh_every: int = 0,   # 0 = off; e.g. 100 = refresh every 100 epochs
    # ---------- Fix 4: Two-stream ----------
    use_two_stream: bool = True,
    n_domain_slots: int = 8,
    n_state_slots: int = 8,
    lambda_usage_domain: float = 0.02,
    lambda_usage_state: float = 0.02,
    # ---------- Fix 5: Entropy gate ----------
    entropy_gate: bool = True,
):
    """Train the enhanced memory model with up to 5 architectural fixes.

    Parameters mirror train_spatial_address_model where equivalent.
    New parameters:

    lambda_repulsion > 0
        Fix 1. Added to loss as +lambda_repulsion * key_repulsion_loss(keys).
        Explicitly minimises mean off-diagonal cosine similarity of memory keys,
        closing the blind spot where usage_entropy cannot detect key collapse.

    lambda_kl_contrastive > 0
        Fix 2. Two augmented views (adj edge dropout + gene masking) are
        passed through the model; symmetric KL divergence between their address
        distributions is minimised. Unlike the existing permutation-based
        contrastive_address_loss, both views retain spatial structure so the
        learned invariance is to mild noise, not global shuffle.

    adj_refresh_every > 0
        Fix 3. Recomputes expression-weighted adjacency every N epochs from the
        model's own reconstructions (cleaner signal as training progresses).
        Off by default (expensive; benchmark as a separate ablation).

    use_two_stream = True
        Fix 4. Uses TwoStreamMemoryAutoencoder instead of
        SpatialAddressMemoryAutoencoder. Each stream has its own
        usage_entropy regulariser (lambda_usage_domain / lambda_usage_state).
        When False, falls back to the original single-stream model with all
        the same extra loss terms (lambda_repulsion, lambda_kl_contrastive).

    entropy_gate = True
        Fix 5 (only active when use_two_stream=True, applied to domain stream).
        Propagation depth is gated by per-spot address certainty; high-entropy
        (uncertain) spots blend back toward their own initial address instead
        of propagating n_hops unconditionally.

    Returns
    -------
    model     : trained nn.Module
    adata     : AnnData with embedding in obsm["X_enhanced"]
    history   : list of per-log-point diagnostic dicts
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hvg_features = get_hvg_features(adata)
    x = torch.tensor(hvg_features, dtype=torch.float32).to(device)

    if expression_weighted:
        adjacency = expression_weighted_adjacency(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )
    else:
        adjacency = normalized_adjacency(adata.obsp["spatial_connectivities"], device=device)

    # Build model
    feature_dim = x.shape[1]
    if use_two_stream:
        model = TwoStreamMemoryAutoencoder(
            feature_dim=feature_dim,
            n_domain_slots=n_domain_slots,
            n_state_slots=n_state_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
            entropy_gate=entropy_gate,
        ).to(device)
    else:
        # Single-stream fallback: original architecture + extra loss terms
        memory_slots = n_domain_slots + n_state_slots  # keep total the same
        model = SpatialAddressMemoryAutoencoder(
            feature_dim=feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
        ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    for epoch in range(epochs):
        # ---- Fix 3: dynamic adjacency refresh ----------------------------
        if adj_refresh_every > 0 and epoch > 0 and epoch % adj_refresh_every == 0:
            model.eval()
            with torch.no_grad():
                if use_two_stream:
                    recon, _, _, _ = model(x, adjacency)
                else:
                    recon, _, _ = model(x, adjacency)
                recon_np = recon.cpu().numpy()
            adjacency = expression_weighted_adjacency(
                adata.obsp["spatial_connectivities"], recon_np, device=device
            )
            model.train()

        model.train()
        optimizer.zero_grad()

        if use_two_stream:
            reconstruction, embedding, A_domain, A_state = model(x, adjacency)
        else:
            reconstruction, embedding, attn_weights = model(x, adjacency)

        recon_loss = F.mse_loss(reconstruction, x)
        loss = recon_loss

        # ---- Usage entropy regularisation --------------------------------
        if use_two_stream:
            dom_ent = usage_entropy(A_domain)
            st_ent = usage_entropy(A_state)
            if lambda_usage_domain:
                loss = loss - lambda_usage_domain * dom_ent
            if lambda_usage_state:
                loss = loss - lambda_usage_state * st_ent
        else:
            slot_ent = usage_entropy(attn_weights)
            if lambda_usage:
                loss = loss - lambda_usage * slot_ent

        # ---- Fix 1: Key Repulsion ----------------------------------------
        if lambda_repulsion:
            if use_two_stream:
                rep = key_repulsion_loss(model.memory.domain_keys) + \
                      key_repulsion_loss(model.memory.state_keys)
                loss = loss + lambda_repulsion * rep * 0.5
            else:
                loss = loss + lambda_repulsion * key_repulsion_loss(
                    model.memory.memory_keys
                )

        # ---- Fix 2: KL Contrastive ---------------------------------------
        if lambda_kl_contrastive:
            # View 1: edge-dropped adjacency, original features
            adj_aug1 = augment_adjacency(adjacency, drop_rate=adj_drop_rate)
            # View 2: original adjacency, gene-masked features
            x_aug2 = augment_features(x, mask_rate=feat_mask_rate)

            if use_two_stream:
                _, _, A_v1_dom, _ = model(x, adj_aug1)
                _, _, A_v2_dom, _ = model(x_aug2, adjacency)
                kl = kl_contrastive_address_loss(A_v1_dom, A_v2_dom)
            else:
                _, _, A_v1 = model(x, adj_aug1)
                _, _, A_v2 = model(x_aug2, adjacency)
                kl = kl_contrastive_address_loss(A_v1, A_v2)

            loss = loss + lambda_kl_contrastive * kl

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                if use_two_stream:
                    dom_sim = key_cosine_similarity(model.memory.domain_keys).item()
                    st_sim = key_cosine_similarity(model.memory.state_keys).item()
                    dom_used = A_domain.argmax(-1).unique().numel()
                    st_used = A_state.argmax(-1).unique().numel()
                    row = {
                        "epoch": epoch,
                        "recon_loss": recon_loss.item(),
                        "total_loss": loss.item(),
                        "dom_usage_entropy": dom_ent.item(),
                        "state_usage_entropy": st_ent.item(),
                        "dom_key_cosine_sim": dom_sim,
                        "state_key_cosine_sim": st_sim,
                        "dom_slots_used": int(dom_used),
                        "state_slots_used": int(st_used),
                    }
                    log_line = (
                        f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                        f"dom_ent={row['dom_usage_entropy']:.3f}  "
                        f"st_ent={row['state_usage_entropy']:.3f}  "
                        f"dom_cos={dom_sim:.3f}  state_cos={st_sim:.3f}  "
                        f"dom_slots={dom_used}/{n_domain_slots}  "
                        f"state_slots={st_used}/{n_state_slots}"
                    )
                else:
                    key_sim = key_cosine_similarity(model.memory.memory_keys).item()
                    n_slots_used = attn_weights.argmax(-1).unique().numel()
                    total_slots = n_domain_slots + n_state_slots
                    row = {
                        "epoch": epoch,
                        "recon_loss": recon_loss.item(),
                        "total_loss": loss.item(),
                        "usage_entropy": slot_ent.item(),
                        "key_cosine_sim": key_sim,
                        "slots_used": int(n_slots_used),
                    }
                    log_line = (
                        f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                        f"ent={row['usage_entropy']:.3f}  "
                        f"cos={key_sim:.3f}  slots={n_slots_used}/{total_slots}"
                    )
            history.append(row)
            if verbose:
                print(log_line)

    model.eval()
    with torch.no_grad():
        if use_two_stream:
            _, embedding, _, _ = model(x, adjacency)
        else:
            _, embedding, _ = model(x, adjacency)

    adata.obsm["X_enhanced"] = embedding.cpu().numpy()
    return model, adata, history
