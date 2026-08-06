"""Masked Spatial Address Propagation (MSAP) -- novel architecture for RECOMB 2027.

Hypothesis: forcing the model to reconstruct masked spots' expression from
PROPAGATED spatial context eliminates the 0-hop collapse failure mode without
needing an explicit usage-entropy anti-collapse term (though we keep it as a
safety net). The mask forces the only gradient signal for masked spots through
the spatial graph, so the model MUST use neighbours to reconstruct them -- the
same idea as BERT's masked language model, applied to spatial address propagation.

Cross-platform generalization comes from attention pooling over hop depths: a
per-spot softmax over [A_0, A_1, ..., A_max_hops] lets the model dynamically
choose its receptive field per spot. Small domains (breast cancer, 28-190 spots)
should learn to down-weight deep hops; large laminar bands (DLPFC, 166-1000
spots) should up-weight them. This is a fundamentally different approach from
adaptive_hops (a per-spot GATE over fixed hops, which collapsed without
regularization) -- here the attention pooling is conditioned on Q directly and
the loss signal is ZINB on masked spots, not MSE on all spots.

Design decisions that differ from SpatialAddressMemoryLayer:
  * mask_token: learnable stand-in for any masked spot's features (like BERT's
    [MASK] embedding). Without this, a masked spot would receive zero input,
    which is a constant input that the encoder would map to the same query
    regardless of the spot's true neighbourhood -- defeating the purpose.
  * ZINB decoder (CountDecoder) rather than MSE: on 68-97% sparse count
    matrices, MSE on zero-inflated data is the wrong likelihood. Using the same
    CountDecoder that was tested (and REJECTED as a full replacement in Stage 3,
    train_count_model.py) -- here the decoder is correct-but-secondary; the
    NOVEL part is the masking + multi-hop attention pooling that feeds it.
  * memory_slots=16: same cross-validated default as the baseline (Stage 8).
  * max_hops=6: extends the baseline's n_hops=4 to give the attention pooling
    more range to work with, covering both DLPFC's ~60-spot 4-hop reach and
    potentially larger neighbourhoods for big laminar bands.

No existing files are modified. All reused components are imported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.count_losses import CountDecoder
from src.models.memory_layer import address_distribution


class MaskedSpatialAddressLayer(nn.Module):
    """Memory addressing with masking + attention pooling over multi-hop depths.

    forward(x, adjacency, mask=None) -> (embedding, attn_weights, hop_weights)

    x          : (N, feature_dim) HVG features (log-normalized, scaled)
    adjacency  : (N, N) sparse row-stochastic adjacency (from
                 expression_weighted_adjacency or normalized_adjacency)
    mask       : (N,) bool tensor; True = spot is masked (replace with mask_token).
                 If None, no masking -- used at inference time.

    Returns:
        embedding   : (N, memory_dim)
        attn_weights: (N, memory_slots) -- the fused address distribution A_fused.
                      Used by usage_entropy / attention_entropy diagnostics, same
                      interface as SpatialAddressMemoryLayer.
        hop_weights : (N, max_hops+1) -- per-spot softmax weights over hop depths.
                      Logged to diagnose whether small-domain datasets really use
                      fewer hops (the core generalization hypothesis).
    """

    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        max_hops=6,
        temperature=1.0,
        attention_fn="softmax",
    ):
        super().__init__()
        if max_hops < 0:
            raise ValueError(f"max_hops must be >= 0, got {max_hops}")

        self.feature_dim = feature_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.max_hops = max_hops
        self.temperature = temperature
        self.attention_fn = attention_fn

        # Learnable stand-in for masked spots.  Initialised to zero so that at
        # epoch 0 masked spots produce the same query as a true zero-expression
        # spot; the encoder quickly learns to distinguish the token from genuine
        # data signal as gradients flow through it.
        self.mask_token = nn.Parameter(torch.zeros(feature_dim))

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)

        # Stored for external diagnostics -- never used inside the forward pass
        # itself to avoid confusion about what is a return value vs. side effect.
        self.last_hop_weights = None

    def forward(self, x, adjacency, mask=None):
        # -- 1. Masking --
        if mask is not None and mask.any():
            x_in = x.clone()
            x_in[mask] = self.mask_token.to(dtype=x.dtype)
        else:
            x_in = x

        # -- 2. Encode to query space --
        queries = self.encoder(x_in)  # (N, memory_dim)

        # -- 3. Base address distribution A_0 --
        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        A0 = address_distribution(attn_scores, self.attention_fn, dim=-1)  # (N, M)

        # -- 4. Build multi-hop address stack [A_0, ..., A_max_hops] --
        addr_by_hop = [A0]
        current = A0
        if adjacency is not None:
            for _ in range(self.max_hops):
                current = torch.sparse.mm(adjacency, current)
                addr_by_hop.append(current)
        addr_stack = torch.stack(addr_by_hop, dim=1)  # (N, max_hops+1, M)

        # -- 5. Attention pooling: Q attends to its own multi-hop addresses --
        # For each hop k, score = A_k[i] · q_proj[i], where q_proj[i] maps Q
        # back to slot space via the same memory_keys used to build A_0.
        # This means "how much does hop k's propagated address distribution
        # agree with the raw (unpropagated) slot affinities of this spot?"
        # Spots in homogeneous domains should prefer deep hops (propagated
        # addresses are already sharp and reinforce self); spots at boundaries
        # or in small domains should prefer shallow hops.
        #
        # Shapes:
        #   queries      : (N, memory_dim=D)
        #   memory_keys  : (memory_slots=M, D)
        #   q_proj       : (N, M)   -- raw slot affinities (same as A_0 scores)
        #   addr_stack   : (N, max_hops+1, M)
        #   hop_scores   : (N, max_hops+1)  via bmm: (N,H,M)@(N,M,1) -> (N,H)
        q_proj = torch.matmul(queries, self.memory_keys.T)  # (N, M)
        hop_scores = torch.bmm(
            addr_stack,                         # (N, max_hops+1, M)
            q_proj.unsqueeze(-1),               # (N, M, 1)
        ).squeeze(-1)                           # (N, max_hops+1)
        hop_weights = F.softmax(hop_scores, dim=-1)  # (N, max_hops+1)

        # Convex combination of simplices stays a valid simplex (no
        # renormalisation needed, same invariant as the fixed-hop path).
        A_fused = (hop_weights.unsqueeze(-1) * addr_stack).sum(dim=1)  # (N, M)

        self.last_hop_weights = hop_weights  # stored for logging, not for gradient

        # -- 6. Read out embedding --
        embedding = torch.matmul(A_fused, self.memory_values)  # (N, memory_dim)
        return embedding, A_fused, hop_weights


class MaskedSpatialAddressAutoencoder(nn.Module):
    """MaskedSpatialAddressLayer + CountDecoder (ZINB reconstruction).

    forward(x, adjacency, library_size, mask=None)
        -> (mu, theta, pi_logits), embedding, attn_weights

    library_size : (N,) or (N, 1) float tensor -- spot-level sequencing depth.
                   Used by CountDecoder to scale the predicted mean proportion.
    mask         : (N,) bool tensor or None.  None at inference (no masking).

    The loss is computed ONLY on masked spots by the training loop:
        zinb_loss(x_counts[mask], mu[mask], theta[mask], pi_logits[mask])
    This file does not call the loss -- that responsibility belongs to
    train_msap_model.py, exactly as train_spatial_address.py owns the MSE loss.
    """

    def __init__(
        self,
        feature_dim,
        n_genes,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        max_hops=6,
        temperature=1.0,
        attention_fn="softmax",
        zero_inflated=True,
    ):
        super().__init__()
        self.memory = MaskedSpatialAddressLayer(
            feature_dim=feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            max_hops=max_hops,
            temperature=temperature,
            attention_fn=attention_fn,
        )
        self.decoder = CountDecoder(
            embedding_dim=memory_dim,
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            zero_inflated=zero_inflated,
        )

    def forward(self, x, adjacency, library_size, mask=None):
        embedding, attn_weights, hop_weights = self.memory(x, adjacency, mask)
        # library_size must be broadcastable to (N, 1) for CountDecoder
        lib = library_size.reshape(-1, 1)
        mu, theta, pi_logits = self.decoder(embedding, lib)
        return (mu, theta, pi_logits), embedding, attn_weights
