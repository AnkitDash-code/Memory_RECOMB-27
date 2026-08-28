"""Training loop for LossFreeGatedMemoryAutoencoder (Stage 3).

Deliberately NO auxiliary loss term for balancing -- the only thing keeping
depth-bucket usage from collapsing is model.update_bias(), called after
every optimizer.step(), never inside the loss.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.preprocess import get_hvg_features
from src.models.loss_free_gated_layer import LossFreeGatedMemoryAutoencoder
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


def train_loss_free_gated_model(
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
    bias_update_rate=0.01,
    seed=0,
    device=None,
    log_every=100,
    verbose=True,
):
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

    feature_dim = x.shape[1]
    model = LossFreeGatedMemoryAutoencoder(
        feature_dim=feature_dim,
        memory_slots=memory_slots,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        n_hops=n_hops,
        temperature=temperature,
        attention_fn=attention_fn,
        bias_update_rate=bias_update_rate,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstruction, embedding, attn_weights, gate_weights = model(x, adjacency)
        recon_loss = F.mse_loss(reconstruction, x)
        slot_ent = usage_entropy(attn_weights)
        # NOTE: no term derived from gate_weights in the loss -- balancing is
        # entirely the bias-update rule below, not backprop.
        loss = recon_loss - lambda_usage * slot_ent

        loss.backward()
        optimizer.step()
        model.update_bias()  # non-differentiable, after the optimizer step

        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                key_sim = key_cosine_similarity(model.memory.memory_keys).item()
                n_slots_used = attn_weights.argmax(-1).unique().numel()
                mean_gate = gate_weights.mean(dim=0)  # (n_hops+1,) usage per depth bucket
                depth_used = (mean_gate > (0.5 / (n_hops + 1))).sum().item()
                row = {
                    "epoch": epoch,
                    "recon_loss": recon_loss.item(),
                    "total_loss": loss.item(),
                    "usage_entropy": slot_ent.item(),
                    "max_entropy": float(np.log(memory_slots)),
                    "key_cosine_similarity": key_sim,
                    "n_slots_used": int(n_slots_used),
                    "mean_gate_by_depth": [float(v) for v in mean_gate.tolist()],
                    "depth_buckets_used": int(depth_used),
                    "depth_bias": [float(v) for v in model.memory.depth_bias.tolist()],
                }
            history.append(row)
            if verbose:
                gate_str = ", ".join(f"{v:.3f}" for v in row["mean_gate_by_depth"])
                print(
                    f"epoch {epoch:4d}  recon={row['recon_loss']:.4f}  "
                    f"ent={row['usage_entropy']:.3f}  cos={key_sim:.3f}  "
                    f"slots={n_slots_used}/{memory_slots}  gate=[{gate_str}]"
                )

    model.eval()
    with torch.no_grad():
        _, embedding, _, _ = model(x, adjacency)

    adata.obsm["X_loss_free_gated"] = embedding.cpu().numpy()
    return model, adata, history
