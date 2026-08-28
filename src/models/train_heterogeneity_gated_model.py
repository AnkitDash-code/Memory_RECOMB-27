"""Training loop for HeterogeneityGatedMemoryAutoencoder (Stage 1).

The certainty score is computed ONCE, before the training loop starts, from
the raw (untrained) HVG features and the spatial graph -- it never sees a
gradient and is recomputed only if this function is called again (e.g. for a
different seed), never inside the loop. This is the entire point of the
design: nothing about the gate can be influenced by what the model learns.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.physical_scale import get_average_edge_length_um
from src.data.preprocess import get_hvg_features
from src.models.heterogeneity_gated_layer import (
    HeterogeneityGatedMemoryAutoencoder,
    compute_fixed_certainty,
)
from src.models.memory_layer import (
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


def train_heterogeneity_gated_model(
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
    platform="visium",
    reference_max_hops=4,
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
    """Train the heterogeneity-gated model. Mirrors train_spatial_address_model's
    loop structure exactly (same loss terms, same logging convention) so any
    ARI difference is attributable to the propagation-gating change alone,
    not a different training recipe.
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

    # Fixed certainty: computed once, from raw data, before any training step.
    avg_edge_length_um = get_average_edge_length_um(adata, platform)
    physical_radius_um = float(reference_max_hops) * avg_edge_length_um
    certainty_np = compute_fixed_certainty(
        hvg_features,
        adata.obsp["spatial_connectivities"],
        physical_radius_um,
        platform,
        avg_edge_length_um,
    )
    certainty = torch.tensor(certainty_np, dtype=torch.float32, device=device)
    assert not certainty.requires_grad

    feature_dim = x.shape[1]
    model = HeterogeneityGatedMemoryAutoencoder(
        feature_dim=feature_dim,
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        temperature=temperature,
        attention_fn=attention_fn,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstruction, embedding, attn_weights = model(x, adjacency, certainty)
        recon_loss = F.mse_loss(reconstruction, x)
        slot_ent = usage_entropy(attn_weights)
        loss = recon_loss - lambda_usage * slot_ent

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                key_sim = key_cosine_similarity(model.memory.memory_keys).item()
                n_slots_used = attn_weights.argmax(-1).unique().numel()
                median_entropy = (
                    -(attn_weights * (attn_weights + 1e-12).log()).sum(-1)
                ).median().item()
                row = {
                    "epoch": epoch,
                    "recon_loss": recon_loss.item(),
                    "total_loss": loss.item(),
                    "usage_entropy": slot_ent.item(),
                    "median_entropy": median_entropy,
                    "max_entropy": float(np.log(memory_slots)),
                    "key_cosine_similarity": key_sim,
                    "n_slots_used": int(n_slots_used),
                    "certainty_mean": float(certainty.mean().item()),
                    "certainty_std": float(certainty.std().item()),
                }
            history.append(row)
            if verbose:
                print(
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"ent={row['usage_entropy']:.3f}  cos={key_sim:.3f}  "
                    f"slots={n_slots_used}/{memory_slots}  "
                    f"certainty_mean={row['certainty_mean']:.3f}"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, _ = model(x, adjacency, certainty)

    adata.obsm["X_heterogeneity_gated"] = embedding.cpu().numpy()
    return model, adata, history
