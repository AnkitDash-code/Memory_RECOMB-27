"""Topologically-Ordered Memory (TOM): give the memory bank a 1D geometry.

Every mechanism tested so far in this project treats memory slots as an
*unordered* bag of prototypes -- nothing says slot 3 is "between" slot 2 and
slot 4. But the ground truth is ordered: cortical layers form a fixed linear
sequence (L1 -> L2/3 -> L4 -> L5 -> L6 -> WM), and the failure mode that has
survived every fix (Stages 8, 9, 11, 13, 14, 15, and the falsified
multimodal attempt in 16/17) is exactly where an unordered address space has
no way to prefer "adjacent layer" over "completely different layer" for an
ambiguous spot.

Built on the differentiable SOM mechanism from SOM-VAE (Fortuin et al.,
ICLR 2019, https://openreview.net/pdf?id=rygjcsR9Y7). Literature check before
implementing: SOMs have been applied to spatial transcriptomics (SOMDE,
Bioinformatics 2021) but for spatially-variable-*gene* identification, not
domain clustering with ordered slots; every DLPFC domain method surveyed
(GraphST, STAGATE, DeepST, SemanticST, SpaBatch, SEDR, SpaGCN, BayesSpace,
stLearn) uses an unordered cluster space. Cortical depth *is* used as an
ordering axis in the field, but for annotation/cell assignment rather than as
an architectural prior inside an unsupervised model -- so the biological
premise is well-supported and the novelty claim is specifically about the
mechanism, not the biology.

Subclasses SpatialAddressMemoryLayer rather than replacing it: the validated
address-propagation + expression-weighted adjacency (Stage 13) stays exactly
as-is, and the topological machinery is layered additively on top, so its
individual contribution can be ablated.

TWO DELIBERATE DEVIATIONS from the source plan's code sketch, both to fix
real problems rather than by preference:

1. SOM kernel distances use INTEGER slot indices (0, 1, ..., n-1), not
   `linspace(0, 1, n)`. With linspace, the maximum possible slot distance is
   1.0, so the plan's suggested `som_sigma` sweep (0.5-2.5) would make
   `exp(-d^2 / 2*sigma^2) >= 0.80` for *every* slot pair -- an almost
   perfectly flat kernel that pulls every slot toward every input. That is
   precisely the "too large and the whole map homogenizes" degeneracy the
   plan's own Section 3 warns about, and it would have silently made the
   whole mechanism a no-op. Integer indices are also the standard SOM
   convention, under which sigma is read as "roughly this many slots away"
   and the suggested sweep values are meaningful.
2. `expected_pos` is reported on a NORMALIZED [0, 1] scale (index /
   (n_slots - 1)). This keeps `lambda_ordinal` independent of `memory_slots`
   -- otherwise the ordinal loss magnitude would scale with slot count and
   any tuned lambda would silently break when memory_slots changes (a live
   risk, since the plan explicitly re-opens memory_slots as a hyperparameter).
"""

import torch
import torch.nn as nn

from src.models.memory_layer import SpatialAddressMemoryLayer


class TopologicalMemoryLayer(SpatialAddressMemoryLayer):
    """SpatialAddressMemoryLayer whose slots carry a 1D topological order."""

    def __init__(self, *args, som_sigma=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.som_sigma = som_sigma
        n_slots = self.memory_keys.shape[0]
        # Integer positions drive the SOM kernel (see deviation 1 above).
        self.register_buffer("slot_index", torch.arange(n_slots, dtype=torch.float32))
        # Normalized positions are what gets reported/penalized (deviation 2).
        denom = max(n_slots - 1, 1)
        self.register_buffer("slot_pos", self.slot_index / denom)

    def _som_neighbor_kernel(self):
        """(n_slots, n_slots) Gaussian over slot-INDEX distance."""
        diff = self.slot_index.unsqueeze(0) - self.slot_index.unsqueeze(1)
        return torch.exp(-(diff**2) / (2 * self.som_sigma**2))

    def expected_position(self, attn_weights):
        """Per-spot expected slot position on the normalized [0, 1] axis.

        This is the key new diagnostic object: if the learned 1D axis is
        tracking anything real, this should correlate with true cortical
        depth order, checkable independently of final ARI.
        """
        return torch.matmul(attn_weights, self.slot_pos)

    def som_topology_loss(self, queries, attn_weights):
        """SOM-VAE-style topology term.

        Pulls each slot's key toward inputs weighted by how close that slot is
        (in slot-index space) to the input's winning slot -- not just the
        winner itself. Without this term the slot ordering is arbitrary and
        the entire TOM idea is a no-op, so it is the load-bearing piece.

        Note: distances are computed against `queries` (the encoder output),
        not raw `x` as in the plan's sketch. In this architecture
        `memory_keys` live in the encoder's `memory_dim` output space, not raw
        feature space -- using raw x would be a dimension mismatch and,
        worse, would silently compare vectors in unrelated spaces if the dims
        happened to align.
        """
        winner_idx = attn_weights.argmax(dim=-1)
        kernel = self._som_neighbor_kernel()
        neighbor_weight = kernel[winner_idx]  # (n_spots, n_slots)
        dists = torch.cdist(queries, self.memory_keys) ** 2  # (n_spots, n_slots)
        return (neighbor_weight * dists).mean()

    def ordinal_smoothness_loss(self, expected_pos, adjacency):
        """Penalize large jumps in expected slot position between spatial neighbors.

        Operates on the scalar ordinal position rather than the address
        vector, so -- unlike the Stage 15 address-space contrastive term --
        it cannot be satisfied by two dissimilar addresses that happen to sit
        close together in address space. This is the piece that should
        specifically help blurry laminar boundaries.

        The adjacency here carries self-loops (D^-1(A+I)); those entries
        contribute exactly zero to pos_diff, so they are harmless and simply
        dilute the mean slightly.
        """
        adjacency = adjacency.coalesce()
        indices = adjacency.indices()
        rows, cols = indices[0], indices[1]
        weights = adjacency.values()
        pos_diff = expected_pos[rows] - expected_pos[cols]
        return (weights * pos_diff**2).mean()

    def forward(self, x, adjacency=None):
        """Same propagation as the parent, but also returns the internals the
        topology losses need.

        Returns (embedding, propagated_attn, pre_attn, queries):
          * propagated_attn -- post-propagation address, determines the
            embedding and therefore the clustering; used for expected_pos.
          * pre_attn -- the direct per-spot query->key assignment, which is
            what "winning slot" means in SOM terms; used for the SOM loss.
        Keeping these separate matters: using the propagated address for the
        SOM winner would entangle the codebook geometry with spatial
        smoothing and make the ablation uninterpretable.
        """
        if adjacency is not None and self.feature_hops:
            for _ in range(self.feature_hops):
                x = torch.sparse.mm(adjacency, x)

        queries = self.encoder(x)

        if adjacency is not None and self.latent_hops:
            for _ in range(self.latent_hops):
                queries = torch.sparse.mm(adjacency, queries)

        from src.models.memory_layer import address_distribution

        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        pre_attn = address_distribution(attn_scores, self.attention_fn, dim=-1)

        propagated = pre_attn
        if adjacency is not None:
            for _ in range(self.n_hops):
                propagated = torch.sparse.mm(adjacency, propagated)

        embedding = torch.matmul(propagated, self.memory_values)
        return embedding, propagated, pre_attn, queries


class TopologicalMemoryAutoencoder(nn.Module):
    """TopologicalMemoryLayer + decoder back to gene-expression space.

    Mirrors SpatialAddressMemoryAutoencoder exactly except for the extra
    internals returned, so the two are drop-in comparable in the harness.
    """

    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
        feature_hops=0,
        latent_hops=0,
        attention_fn="softmax",
        som_sigma=1.5,
    ):
        super().__init__()
        self.memory = TopologicalMemoryLayer(
            feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
            feature_hops=feature_hops,
            latent_hops=latent_hops,
            attention_fn=attention_fn,
            som_sigma=som_sigma,
        )
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x, adjacency=None):
        embedding, propagated, pre_attn, queries = self.memory(x, adjacency)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, propagated, pre_attn, queries
