"""Training loop for TopologicalMemoryAutoencoder (TOM).

Mirrors train_spatial_address.py exactly -- same defaults, same
expression-weighted adjacency (Stage 13), same usage-entropy anti-collapse
term -- and adds the two topology losses as *auxiliary* terms on top, so the
existing validated configuration is recoverable by setting both lambdas to 0.

Instrumentation is mandatory here rather than optional, following the Stage 3
lesson (a negative result whose cause was never isolated, because the run
wasn't logging enough to tell competing explanations apart) and the Stage
14/15 lesson (single-seed results that looked promising and evaporated across
seeds). Logged from epoch 1 of every run:

  * expected_pos mean/std/min/max -- is the ordinal axis being used at all,
    or has it collapsed to a single position?
  * n_slots_used + usage_entropy -- SOM collapse is a documented failure mode
    in the SOM-VAE literature itself, not hypothetical; the topology loss
    does NOT automatically prevent it, so it is verified, not assumed.
  * key_cosine_similarity -- the quiet codebook-degeneracy mode where slots
    stay "used" but their key vectors become near-duplicates.
  * both topology loss terms separately -- so an ablation can attribute any
    effect to the right piece.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.models.memory_layer import (
    attention_entropy,
    expression_weighted_adjacency,
    key_cosine_similarity,
    normalized_adjacency,
    usage_entropy,
)
from src.models.topological_memory_layer import TopologicalMemoryAutoencoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_topological_model(
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
    lambda_som=0.02,
    lambda_ordinal=0.02,
    som_sigma=1.5,
    expression_weighted=True,
    attention_fn="softmax",
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train the topologically-ordered memory model on preprocessed (HVG) AnnData.

    Defaults for memory_slots / n_hops / lambda_usage / expression_weighted are
    inherited unchanged from the validated Stage 13 configuration so that TOM
    is a strictly additive change. lambda_som and lambda_ordinal start small
    (0.02): these are structural priors meant to shape the codebook, not to
    dominate reconstruction -- the same "auxiliary term" discipline that
    worked for Stage 13 and correctly rejected Stage 15.

    NOTE: memory_slots=16 was cross-validated (Stage 8) for an *unordered*
    codebook. Once slots carry topological meaning the optimum may differ, so
    it should be treated as re-opened rather than settled -- see the source
    plan's Section 5 risk note.

    Writes back to adata:
      obsm['X_topological_address'] -- the embedding, for clustering
      obs['expected_slot_pos']      -- per-spot ordinal position in [0, 1],
                                       the key diagnostic for whether the
                                       learned axis tracks true layer order
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

    model = TopologicalMemoryAutoencoder(
        feature_dim=x.shape[1],
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        temperature=temperature,
        attention_fn=attention_fn,
        som_sigma=som_sigma,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_entropy = math.log(memory_slots)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstruction, _, propagated, pre_attn, queries = model(x, adjacency)
        recon_loss = F.mse_loss(reconstruction, x)

        loss = recon_loss
        slot_usage_entropy = usage_entropy(propagated)
        if lambda_usage:
            loss = loss - lambda_usage * slot_usage_entropy

        som_loss = torch.zeros((), device=device)
        ordinal_loss = torch.zeros((), device=device)
        expected_pos = model.memory.expected_position(propagated)

        if lambda_som:
            # Winner comes from the PRE-propagation address: "which prototype
            # is this spot" is a codebook question, not a spatial-smoothing one.
            som_loss = model.memory.som_topology_loss(queries, pre_attn)
            loss = loss + lambda_som * som_loss
        if lambda_ordinal:
            ordinal_loss = model.memory.ordinal_smoothness_loss(expected_pos, adjacency)
            loss = loss + lambda_ordinal * ordinal_loss

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                median_entropy = attention_entropy(propagated).median().item()
                n_slots_used = int(propagated.argmax(dim=-1).unique().numel())
                key_sim = key_cosine_similarity(model.memory.memory_keys).item()
                pos = expected_pos.detach()
            row = {
                "epoch": epoch,
                "recon_loss": recon_loss.item(),
                "total_loss": loss.item(),
                "som_loss": float(som_loss.item()),
                "ordinal_loss": float(ordinal_loss.item()),
                "median_entropy": median_entropy,
                "usage_entropy": slot_usage_entropy.item(),
                "max_entropy": max_entropy,
                "n_slots_used": n_slots_used,
                "key_cosine_similarity": key_sim,
                "expected_pos_mean": float(pos.mean().item()),
                "expected_pos_std": float(pos.std().item()),
                "expected_pos_min": float(pos.min().item()),
                "expected_pos_max": float(pos.max().item()),
            }
            history.append(row)
            if verbose:
                print(
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"som={row['som_loss']:.4f}  ord={row['ordinal_loss']:.5f}  "
                    f"slots={n_slots_used}  key_cos={key_sim:.3f}  "
                    f"pos={row['expected_pos_mean']:.3f}+/-{row['expected_pos_std']:.3f} "
                    f"[{row['expected_pos_min']:.2f},{row['expected_pos_max']:.2f}]"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, propagated, _, _ = model(x, adjacency)
        expected_pos = model.memory.expected_position(propagated)
    adata.obsm["X_topological_address"] = embedding.cpu().numpy()
    adata.obs["expected_slot_pos"] = expected_pos.cpu().numpy()
    return model, adata, history
