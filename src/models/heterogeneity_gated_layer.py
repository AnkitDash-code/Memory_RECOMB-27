"""Stage 1 of the domain-scale-vs-propagation-depth follow-up plan: gate
propagation depth by an EXTERNALLY computed, FIXED heterogeneity statistic
instead of a learned gate.

Every learned depth/entropy gate tried in this project so far collapsed to
the shallowest, easiest-to-reconstruct behavior (Phase D's adaptive_hops:
collapsed to depth 0 without regularization; enhanced_memory_layer.py's
entropy_gated_propagation: derives "certainty" from the model's OWN address
distribution A0, which is itself shaped by the same reconstruction pressure
that causes collapse elsewhere). The common failure mode is that nothing
external forces the gate's behavior -- reconstruction loss always has an
incentive to prefer less smoothing, and any gate whose input is itself a
function of trained parameters can be pulled along with it.

This module follows ClustSIGNAL's design (Panwar, Guo, Zhou, Hicks,
Ghazanfar, bioRxiv 2025.11.30.691081, doi:10.64898/2025.11.30.691081):
compute a local heterogeneity statistic OUTSIDE the training loop, from raw
(untrained) data only, and map it through a FIXED monotone function to a
per-spot propagation weight. Verified against the actual paper (not just a
plan description) before implementing: ClustSIGNAL computes per-cell entropy
from the composition of pre-clustered subclusters in each cell's fixed-size
neighbourhood, then uses that entropy to weight adaptive smoothing -- the
same "precompute, then fix a monotone map" shape used here, adapted to this
project's address-propagation setting using the local expression-dissimilarity
statistic this repo already has (Hop-Fusion's local_expression_heterogeneity,
src/data/physical_scale.py), rather than re-deriving a new statistic from
scratch.

The one thing that must never happen here: `certainty` must not depend on
any nn.Parameter, must not have requires_grad=True, and must be identical
across repeated calls for the same input data -- see
tests/test_heterogeneity_gated_layer.py for the regression tests pinning
exactly this.
"""

import numpy as np
import torch
import torch.nn as nn

from src.data.physical_scale import (
    get_average_edge_length_um,
    local_expression_heterogeneity,
    um_radius_to_hop_count,
)
from src.models.memory_layer import address_distribution


def compute_fixed_certainty(
    hvg_features: np.ndarray,
    connectivities,
    physical_radius_um: float,
    platform: str,
    avg_edge_length_um: float,
) -> np.ndarray:
    """Precompute a per-spot certainty score in [0, 1], no trainable parameters.

    Reuses local_expression_heterogeneity (already implemented and tested for
    Hop-Fusion) rather than re-deriving a local-dissimilarity statistic from
    scratch. Converts it to a rank-based percentile (not a raw min-max
    normalization) specifically so the resulting certainty scale is
    comparable across datasets with very different absolute expression-
    distance magnitudes (DLPFC vs. breast cancer) -- min-max would make the
    single most-heterogeneous spot on ANY dataset get certainty near 0
    regardless of how heterogeneous the tissue is overall, which is not the
    intended semantics.

    certainty = 1 - percentile_rank(heterogeneity): high local dissimilarity
    (this spot's neighbourhood looks compositionally mixed) -> low certainty
    -> shallow propagation; low dissimilarity (homogeneous neighbourhood) ->
    high certainty -> deep propagation.

    Returns a plain numpy array. Callers must wrap it as a torch tensor with
    requires_grad=False (the default) before use -- never as an nn.Parameter
    or the output of any layer with trainable weights.
    """
    heterogeneity, _hops = local_expression_heterogeneity(
        hvg_features, connectivities, physical_radius_um, platform, avg_edge_length_um
    )
    ranks = np.argsort(np.argsort(heterogeneity)).astype(np.float64)
    n = len(heterogeneity)
    percentile = ranks / max(n - 1, 1)  # 0 = least heterogeneous, 1 = most
    certainty = 1.0 - percentile
    return certainty.astype(np.float32)


def heterogeneity_gated_propagation(
    A0: torch.Tensor,
    adjacency,
    n_hops: int,
    certainty: torch.Tensor,
) -> torch.Tensor:
    """Blend between A0 (no propagation) and A_nhops (full propagation) using
    an EXTERNALLY supplied, fixed per-spot certainty -- never computed from A0
    or any other model output.

    Same convex-combination formula as enhanced_memory_layer.entropy_gated_
    propagation (output = certainty * A_nhops + (1-certainty) * A0, a convex
    combination of two valid simplices and therefore itself a valid simplex),
    deliberately reusing that already-tested mathematical form. The only
    difference, and the entire point of this function, is where `certainty`
    comes from: here it is a precomputed constant, passed in, never derived
    from the address distribution being gated.
    """
    certainty = certainty.reshape(-1, 1).to(device=A0.device, dtype=A0.dtype).detach()

    A_propagated = A0
    for _ in range(n_hops):
        A_propagated = torch.sparse.mm(adjacency, A_propagated)

    return certainty * A_propagated + (1.0 - certainty) * A0


class HeterogeneityGatedMemoryLayer(nn.Module):
    """SpatialAddressMemoryLayer's encoder/keys/values/addressing, with the
    fixed n_hops propagation step replaced by heterogeneity_gated_propagation.

    Kept as a separate class (not a flag added to SpatialAddressMemoryLayer)
    per this project's established discipline of never modifying the tested
    baseline class for a new ablation -- see enhanced_memory_layer.py and
    HopFusionMemoryLayer for the same pattern.
    """

    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
        attention_fn="softmax",
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.n_hops = n_hops
        self.temperature = temperature
        self.attention_fn = attention_fn

    def forward(self, x, adjacency, certainty):
        queries = self.encoder(x)
        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        A0 = address_distribution(attn_scores, self.attention_fn, dim=-1)

        if adjacency is not None and self.n_hops > 0:
            propagated = heterogeneity_gated_propagation(A0, adjacency, self.n_hops, certainty)
        else:
            propagated = A0

        embedding = torch.matmul(propagated, self.memory_values)
        return embedding, propagated


class HeterogeneityGatedMemoryAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
        attention_fn="softmax",
    ):
        super().__init__()
        self.memory = HeterogeneityGatedMemoryLayer(
            feature_dim=feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
            attention_fn=attention_fn,
        )
        self.decoder = nn.Linear(memory_dim, feature_dim)

    def forward(self, x, adjacency, certainty):
        embedding, addresses = self.memory(x, adjacency, certainty)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, addresses
