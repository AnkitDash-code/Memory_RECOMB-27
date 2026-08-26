"""Enhanced memory layer variants implementing 5 architectural fixes.

These are NEW classes/functions kept ENTIRELY SEPARATE from memory_layer.py so
that the existing model is preserved as an untouched baseline and every fix is
an explicit, measurable ablation -- the same discipline the project has applied
to every previous experiment.

Fixes implemented (see implementation_plan.md for full motivation):

  Fix 1  key_repulsion_loss
    Penalise off-diagonal cosine similarity between memory key vectors.
    usage_entropy cannot detect key collapse because a uniform softmax over
    identical logits is maximum-entropy -- the loss happily rewards collapse.

  Fix 2  kl_contrastive_address_loss / augment_adjacency / augment_features
    GraphCL-style contrastive regularisation in address space. Two augmented
    views (edge dropout + gene masking) are passed through the same model;
    KL divergence between their address distributions forces augmentation-
    invariant biological signal.

  Fix 3  Dynamic adjacency refresh (inside train_enhanced_model)
    Expression-weighted adjacency recomputed every adj_refresh_every epochs
    from the model's own reconstructions.

  Fix 4  TwoStreamMemoryLayer / TwoStreamMemoryAutoencoder
    Separate domain slots (propagated -- spatial continuity) from state slots
    (NOT propagated -- local cell-type identity). One bank encoding both is
    forced to decode blurry averages; splitting decouples the objectives.

  Fix 5  entropy_gated_propagation
    Gate propagation depth by per-spot address certainty. Confident spots
    propagate n_hops; uncertain/noisy spots stay near their own initial
    address. No extra parameters -- fully determined by A0.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.memory_layer import (
    expression_weighted_adjacency,
    usage_entropy,
)


# ---------------------------------------------------------------------------
# Fix 1 -- Key Repulsion Loss
# ---------------------------------------------------------------------------


def key_repulsion_loss(memory_keys: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity between memory key rows.

    Minimising this closes the mathematical blind spot in usage_entropy:
    if all K keys collapse to the same direction every query produces
    uniform softmax (maximum usage entropy) yet the model outputs the same
    distribution for every spot. key_cosine_similarity in memory_layer.py
    already measures this as a diagnostic; this function is the
    gradient-carrying counterpart used during training.
    """
    normed = F.normalize(memory_keys, dim=-1)        # (K, D)
    sim = normed @ normed.T                           # (K, K)
    K = sim.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
    return sim[mask].mean()


# ---------------------------------------------------------------------------
# Fix 2 -- KL Contrastive Address Regularisation
# ---------------------------------------------------------------------------


def kl_contrastive_address_loss(A1: torch.Tensor, A2: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two augmented-view address distributions.

    Unlike contrastive_address_loss (permutation corruption + dot product),
    this version uses graph/feature augmentation views so both carry spatial
    structure, and uses KL divergence which concentrates gradient on confident
    but wrong predictions.
    """
    eps = 1e-12
    kl_12 = F.kl_div((A2 + eps).log(), A1, reduction="batchmean")
    kl_21 = F.kl_div((A1 + eps).log(), A2, reduction="batchmean")
    return 0.5 * (kl_12 + kl_21)


def augment_adjacency(adj_sparse: torch.Tensor, drop_rate: float = 0.10) -> torch.Tensor:
    """Row-renormalized adjacency with drop_rate fraction of off-diagonal edges removed.

    Self-loops are never dropped. After dropping, row sums are recomputed so
    the result is still row-stochastic, preserving the simplex invariant.
    """
    adj_sparse = adj_sparse.coalesce()
    indices = adj_sparse.indices()     # (2, nnz)
    values = adj_sparse.values()       # (nnz,)
    shape = adj_sparse.shape

    is_self = indices[0] == indices[1]
    off_diag_pos = (~is_self).nonzero(as_tuple=True)[0]
    n_off = off_diag_pos.shape[0]
    n_drop = int(n_off * drop_rate)

    keep = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    if n_drop > 0:
        chosen = torch.randperm(n_off, device=values.device)[:n_drop]
        keep[off_diag_pos[chosen]] = False

    kept_idx = indices[:, keep]
    kept_val = values[keep]

    row_sums = torch.zeros(shape[0], dtype=values.dtype, device=values.device)
    row_sums.scatter_add_(0, kept_idx[0], kept_val)
    row_sums.clamp_(min=1e-12)
    norm_val = kept_val / row_sums[kept_idx[0]]

    return torch.sparse_coo_tensor(kept_idx, norm_val, shape).coalesce()


def augment_features(x: torch.Tensor, mask_rate: float = 0.05) -> torch.Tensor:
    """Zero out mask_rate fraction of HVG features per spot independently (gene masking)."""
    mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_rate))
    return x * mask


# ---------------------------------------------------------------------------
# Fix 5 -- Entropy-Gated Propagation
# ---------------------------------------------------------------------------


def entropy_gated_propagation(
    A0: torch.Tensor,
    adjacency: torch.Tensor,
    n_hops: int,
) -> torch.Tensor:
    """Blend between A0 (no propagation) and A_nhops (full propagation) by certainty.

    Confident spots (low entropy of A0) propagate their domain identity widely.
    Uncertain spots (high entropy -- heavy dropout, boundary zones) stay close
    to their own initial address rather than spreading noise.

    Gate formula (no extra parameters):
        H        = Shannon entropy of A0 per spot  (N,) nats
        H_max    = log(K)
        certainty= 1 - H / H_max                  (N,) in [0, 1]
        output   = certainty * A_nhops + (1-certainty) * A0

    The output is a convex combination of two valid simplices and is itself
    a valid simplex (rows sum to 1).

    Contrast with adaptive_hops (Phase D): that variant learns a gate via a
    linear projection of the query, which collapsed to depth 0 for every spot
    without lambda_hop_usage regularisation and scored 0.350 -- worse than
    fixed n_hops=0 (0.391). The entropy gate here collapses only for spots
    that are genuinely confused (the right behaviour), not because the
    optimiser found a reconstruction shortcut.
    """
    K = A0.shape[-1]
    eps = 1e-12
    H = -(A0 * (A0 + eps).log()).sum(-1)                         # (N,) nats
    H_max = math.log(K) if K > 1 else 1.0
    certainty = (1.0 - H / H_max).clamp(0.0, 1.0).unsqueeze(-1) # (N, 1)

    A_propagated = A0
    for _ in range(n_hops):
        A_propagated = torch.sparse.mm(adjacency, A_propagated)

    return certainty * A_propagated + (1.0 - certainty) * A0


# ---------------------------------------------------------------------------
# Fix 4 -- Two-Stream Memory Layer
# ---------------------------------------------------------------------------


class TwoStreamMemoryLayer(nn.Module):
    """Domain slots (propagated) + state slots (not propagated).

    Splitting the memory bank decouples two objectives that compete when
    sharing a single bank:

      domain_keys / domain_values -- propagated n_hops over spatial graph.
        Captures broad spatial continuity: adjacent spots in the same cortical
        layer converge to a shared domain address even when fine-grained
        expression differs (noise, dropout, mixed cell types).

      state_keys / state_values -- NOT propagated.
        Captures local cell-type identity from the spot's own expression.
        Adjacent spots can have different state addresses (neuron vs. glia)
        while sharing the same domain address (Layer 4).

    Embedding dimension = 2 * memory_dim (half domain, half state).
    Each stream has its own usage_entropy loss in the training loop.
    """

    def __init__(
        self,
        feature_dim: int,
        n_domain_slots: int = 8,
        n_state_slots: int = 8,
        memory_dim: int = 128,
        hidden_dim: int = 256,
        n_hops: int = 4,
        temperature: float = 1.0,
        entropy_gate: bool = False,
    ):
        super().__init__()
        self.n_domain_slots = n_domain_slots
        self.n_state_slots = n_state_slots
        self.memory_dim = memory_dim
        self.n_hops = n_hops
        self.temperature = temperature
        self.entropy_gate = entropy_gate

        # Shared encoder -- both streams read the same latent query
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        # Domain stream -- propagated
        self.domain_keys = nn.Parameter(torch.randn(n_domain_slots, memory_dim) * 0.02)
        self.domain_values = nn.Parameter(torch.randn(n_domain_slots, memory_dim) * 0.02)

        # State stream -- not propagated
        self.state_keys = nn.Parameter(torch.randn(n_state_slots, memory_dim) * 0.02)
        self.state_values = nn.Parameter(torch.randn(n_state_slots, memory_dim) * 0.02)

        # Stash for loss computation in training loop
        self.last_domain_addresses: torch.Tensor | None = None
        self.last_state_addresses: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, adjacency=None):
        """
        Returns
        -------
        embedding : (N, 2*memory_dim) -- concat of domain and state readouts
        A_domain  : (N, n_domain_slots) -- propagated domain address distribution
        A_state   : (N, n_state_slots)  -- local state address distribution
        """
        q = self.encoder(x)   # (N, memory_dim)

        # Domain stream
        dom_scores = (q @ self.domain_keys.T) / self.temperature   # (N, K_d)
        A_domain = F.softmax(dom_scores, dim=-1)                    # (N, K_d)

        if adjacency is not None and self.n_hops > 0:
            if self.entropy_gate:
                A_domain = entropy_gated_propagation(A_domain, adjacency, self.n_hops)
            else:
                for _ in range(self.n_hops):
                    A_domain = torch.sparse.mm(adjacency, A_domain)

        z_domain = A_domain @ self.domain_values   # (N, memory_dim)

        # State stream (no propagation)
        st_scores = (q @ self.state_keys.T) / self.temperature   # (N, K_s)
        A_state = F.softmax(st_scores, dim=-1)                    # (N, K_s)
        z_state = A_state @ self.state_values                      # (N, memory_dim)

        embedding = torch.cat([z_domain, z_state], dim=-1)        # (N, 2*memory_dim)

        self.last_domain_addresses = A_domain
        self.last_state_addresses = A_state

        return embedding, A_domain, A_state


class TwoStreamMemoryAutoencoder(nn.Module):
    """Trainable wrapper: TwoStreamMemoryLayer + linear decoder.

    Decoder input is the full 2*memory_dim concatenated embedding so the
    reconstruction gradient reaches both domain and state banks equally.
    """

    def __init__(
        self,
        feature_dim: int,
        n_domain_slots: int = 8,
        n_state_slots: int = 8,
        memory_dim: int = 128,
        hidden_dim: int = 256,
        n_hops: int = 4,
        temperature: float = 1.0,
        entropy_gate: bool = False,
    ):
        super().__init__()
        self.memory = TwoStreamMemoryLayer(
            feature_dim=feature_dim,
            n_domain_slots=n_domain_slots,
            n_state_slots=n_state_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
            entropy_gate=entropy_gate,
        )
        self.decoder = nn.Linear(2 * memory_dim, feature_dim)

    def forward(self, x: torch.Tensor, adjacency=None):
        embedding, A_domain, A_state = self.memory(x, adjacency)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, A_domain, A_state
