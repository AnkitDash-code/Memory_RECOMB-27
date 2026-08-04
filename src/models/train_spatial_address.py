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
    address_spatial_coherence_loss,
    attention_entropy,
    contrastive_address_loss,
    expression_weighted_adjacency,
    expression_similarity_edge_weights,
    key_cosine_similarity,
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
    memory_slots=16,
    memory_dim=128,
    hidden_dim=256,
    n_hops=4,
    temperature=1.0,
    feature_hops=0,
    latent_hops=0,
    lambda_usage=0.02,
    lambda_sharpen=0.0,
    lambda_spatial_coherence=0.0,
    lambda_contrastive=0.0,
    kmeans_init=False,
    expression_weighted=True,
    attention_fn="softmax",
    adaptive_hops=False,
    lambda_hop_usage=0.0,
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train the address-propagation model on a preprocessed (HVG) AnnData.

    lambda_usage > 0 maximizes the entropy of MARGINAL slot usage, which is what
    stops slot collapse (see usage_entropy). With it at 0 the model reliably
    collapses to a single slot decoding the dataset mean (measured:
    slots_used=1, ARI=0.0), so it defaults on rather than off.

    lambda_sharpen > 0 additionally *minimizes* per-row entropy, pushing each
    individual spot to commit to a slot -- pulls in a complementary direction
    to lambda_usage (spread usage across the codebook, but keep each spot's
    own assignment confident).

    kmeans_init=True replaces the random memory_keys init with k-means
    centroids of the (still-randomly-weighted) encoder's queries on the real
    data -- standard practice in VQ-style codebook methods, tested here as a
    candidate fix for the documented high per-seed ARI variance (see
    SpatialAddressMemoryLayer.initialize_keys_kmeans). Off by default so its
    effect is an explicit, measured ablation, not an assumed improvement.

    memory_slots=16 is a CROSS-VALIDATED choice (3 held-out slices, none of them
    151673), not a single-slice-tuned one. An earlier sweep on 151673 alone picked
    memory_slots=32 (0.5713 there) -- but on a true held-out set (8 slices used
    in neither the original tuning nor this cross-validation), memory_slots=32
    scored only 0.4601 while memory_slots=16 reached 0.5025. Single-slice
    hyperparameter tuning had overfit to 151673's idiosyncrasies; see
    outputs/logs/stage2_progress.md (Stage 8) for the full CV sweep and both
    held-out comparisons.

    n_hops=4 and lambda_usage=0.02 are now BOTH cross-validated the same way
    memory_slots was (src/eval/cross_validate_hops_usage.py, coordinate descent
    over the same 3 CV-validation slices, checked on the same 8 true held-out
    slices). n_hops=4 confirmed the original single-slice choice -- no change.
    lambda_usage did not: the original single-slice value was 0.1, but 0.02
    scored higher on both the CV slices (0.531 vs 0.467) and, more importantly,
    on the true held-out set (0.520 +/- 0.072 vs 0.503 +/- 0.089) -- a real
    +0.017 ARI gain with LOWER variance, not just a better mean. See
    outputs/logs/stage2_progress.md (Stage 11) for the full sweep.

    feature_hops/latent_hops default to 0, the pure formulation. Both hybrid
    variants were tested and lost to it -- see outputs/logs/stage2_progress.md.

    expression_weighted=True (NEW DEFAULT, Stage 13) reweights each structural
    spatial edge by exp(-||x_i - x_j||^2 / 2*sigma^2) (median-heuristic sigma),
    so address mass propagates less across spatially-adjacent but
    transcriptionally dissimilar spots -- a targeted fix for blurred layer
    boundaries, motivated by the persistent subject-3 gap. Verified first on
    the subject-3 slices alone (all 3 improved on consensus), then confirmed
    at full 12-slice/5-seed scale: held-out consensus 0.549 -> 0.562, per-seed
    0.520 -> 0.534, with LOWER variance on both. It also moved the paired
    significance test against GraphST from "significant" (p=0.042, per-seed)
    to "not significant" (p=0.123) -- see outputs/logs/stage2_progress.md
    (Stage 13) for the full result and an honest note on why this isn't a
    blind held-out validation the way memory_slots/n_hops/lambda_usage's
    cross-validation was (the full-scale number was seen before deciding to
    keep it).

    attention_fn selects how raw address scores are mapped to a probability
    simplex: "softmax" (default), "entmax15", or "sparsemax" (see
    memory_layer.address_distribution). Both sparse alternatives were tested
    (Stage 14) on the subject-3 slices and REJECTED: entmax15 regressed
    clearly on every slice, sparsemax was a wash. "softmax" remains the
    default.

    lambda_contrastive > 0 adds contrastive_address_loss on top of the MSE
    objective: a spot's address under real features is pushed to disagree
    with its address when features are shuffled across spots (permutation
    corruption), discriminating in ADDRESS space specifically. This loss term
    already existed (Stage 3, train_count_model.py) but was only ever tested
    bundled with an NB/ZINB likelihood, and rejected as part of that
    combination without isolating which change caused the regression -- an
    external review's hypothesis for why (contrastive loss applied to
    continuous embeddings rather than discrete addresses) turned out to be
    factually wrong, since it was already address-space; this parameter lets
    it be tested on its own, on top of the winning MSE model. Off by default,
    an explicit ablation. See key_cosine_similarity for the accompanying
    codebook-collapse diagnostic, logged whenever lambda_contrastive != 0.

    adaptive_hops=True (Phase D, see cross_validate_adaptive_hops.py) replaces
    the single fixed n_hops with a per-spot learned gate over depths 0..n_hops
    -- see SpatialAddressMemoryLayer's docstring for the mechanistic finding
    that motivated it (n_hops=4 over-smooths domains smaller than a 4-hop
    neighbourhood, which DLPFC's layers never are but breast cancer's
    fine-grained regions often are). Off by default, an explicit ablation
    until cross-validated the same leakage-safe way memory_slots/n_hops/
    lambda_usage were.

    WITHOUT lambda_hop_usage, adaptive_hops collapses: reconstruction MSE has
    no incentive to use propagation at all (unsmoothed data always reconstructs
    itself more easily), so the gate routes essentially every spot to depth 0
    regardless of domain size -- measured directly (mean gate weight on depth
    0 > 0.999 across 3 held-out slices x 3 seeds), and this collapsed variant
    scored WORSE (0.350 +/- 0.106) than even a fixed n_hops=0 (0.391 +/- 0.114),
    both well below the current default's 0.504 +/- 0.084
    (outputs/logs/adaptive_hops_check_results.json). lambda_hop_usage > 0
    reuses `usage_entropy` (already proven against the analogous slot-collapse
    failure) on the gate weights instead, maximizing the entropy of the
    MARGINAL depth-usage distribution -- forcing the gate to actually spread
    usage across depths in aggregate, the same load-balancing fix applied to
    a different collapse mode of the same underlying problem.
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

    coherence_edge_index = coherence_edge_weight = None
    if lambda_spatial_coherence:
        coherence_edge_index, coherence_edge_weight = expression_similarity_edge_weights(
            adata.obsp["spatial_connectivities"], hvg_features, device=device
        )

    model = SpatialAddressMemoryAutoencoder(
        feature_dim=x.shape[1],
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        temperature=temperature,
        feature_hops=feature_hops,
        latent_hops=latent_hops,
        attention_fn=attention_fn,
        adaptive_hops=adaptive_hops,
    ).to(device)

    if kmeans_init:
        with torch.no_grad():
            initial_queries = model.memory.encoder(x)
        model.memory.initialize_keys_kmeans(initial_queries, seed=seed)

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
        if lambda_spatial_coherence:
            loss = loss + lambda_spatial_coherence * address_spatial_coherence_loss(
                attn_weights, coherence_edge_index, coherence_edge_weight
            )
        if lambda_contrastive:
            permutation = torch.randperm(x.shape[0], device=device)
            _, _, attn_corrupted = model(x[permutation], adjacency)
            loss = loss + lambda_contrastive * contrastive_address_loss(attn_weights, attn_corrupted)
        if adaptive_hops and lambda_hop_usage:
            # Same load-balancing fix as lambda_usage, applied to the hop-depth
            # gate instead of the slot address -- without it the gate collapses
            # to depth 0 for every spot (see train_spatial_address_model's
            # docstring for the measured collapse).
            hop_usage_entropy = usage_entropy(model.memory.last_hop_gate_weights)
            loss = loss - lambda_hop_usage * hop_usage_entropy

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                median_entropy = attention_entropy(attn_weights).median().item()
                n_slots_used = (attn_weights.argmax(dim=-1).unique()).numel()
                key_sim = key_cosine_similarity(model.memory.memory_keys).item()
            row = {
                "epoch": epoch,
                "recon_loss": recon_loss.item(),
                "total_loss": loss.item(),
                "median_entropy": median_entropy,
                "usage_entropy": slot_usage_entropy.item(),
                "max_entropy": max_entropy,
                "n_slots_used": int(n_slots_used),
                "key_cosine_similarity": key_sim,
            }
            log_line = (
                f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                f"row_entropy={median_entropy:.3f}  "
                f"usage_entropy={row['usage_entropy']:.3f}/{max_entropy:.3f}  "
                f"slots_used={n_slots_used}  key_cos_sim={key_sim:.3f}"
            )
            if adaptive_hops:
                with torch.no_grad():
                    gate = model.memory.last_hop_gate_weights
                    mean_gate = gate.mean(dim=0)
                    eff_depth = (mean_gate * torch.arange(mean_gate.shape[0], device=mean_gate.device)).sum().item()
                row["hop_gate_usage_entropy"] = hop_usage_entropy.item() if lambda_hop_usage else None
                row["hop_gate_mean_weights"] = mean_gate.tolist()
                row["hop_gate_effective_depth"] = eff_depth
                log_line += f"  eff_hop_depth={eff_depth:.3f}"
            history.append(row)
            if verbose:
                print(log_line)

    model.eval()
    with torch.no_grad():
        _, embedding, attn_weights = model(x, adjacency)

    adata.obsm["X_spatial_address"] = embedding.cpu().numpy()
    return model, adata, history
