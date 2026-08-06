"""Boundary-Aware Adaptive Propagation (BAAP) -- novel architecture for RECOMB 2027.

Hypothesis: Over-smoothing across tissue domain boundaries occurs when spatial
propagation repeatedly mixes address distributions between neighboring spots that
belong to distinct biological regions.

BAAP solves this by introducing dynamic address-similarity edge gating:
At each propagation hop k, the similarity between the address distributions of
neighboring spots (row, col) is evaluated:
    dist_sq = || A^{(k-1)}_row - A^{(k-1)}_col ||^2
    G = exp( - dist_sq )
The structural graph edge weight is dynamically gated by G before adding self-loops
and row-normalizing. Neighbors with dissimilar address profiles have their edge
weights suppressed, halting propagation across boundary borders.

Multi-hop address distributions [A_0, A_1, ..., A_max_hops] are fused via per-spot
hop attention pooling conditioned on the spot's query vector Q.

No pre-existing files are modified. Reused components are imported from src.models.memory_layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.memory_layer import address_distribution


class BAAPMemoryLayer(nn.Module):
    """Boundary-Aware Adaptive Propagation layer.

    Parameters
    ----------
    feature_dim : int
        Number of input features per spot (e.g. 3000 HVGs).
    memory_slots : int
        Number of memory slots M (default 16, matching baseline cross-validation).
    memory_dim : int
        Dimension of memory key and value vectors d (default 128).
    hidden_dim : int
        Hidden layer dimension for encoder projection (default 256).
    max_hops : int
        Maximum propagation hop depth (default 6).
    temperature : float
        Softmax scaling factor for query-key matching (default 1.0).
    attention_fn : str
        Address distribution function ("softmax" default).
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

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        self.memory_keys = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)

        # Hop attention gate: maps query Q (memory_dim) to max_hops + 1 logits
        self.hop_attention = nn.Linear(memory_dim, max_hops + 1)
        self.last_hop_weights = None

    def forward(self, x, base_adjacency):
        """Forward pass executing dynamic boundary-gated address propagation.

        x              : (N, feature_dim) float tensor of spot features.
        base_adjacency : Sparse COO tensor (N, N) containing base spatial graph edge weights.

        Returns:
            embedding   : (N, memory_dim)
            A_fused     : (N, memory_slots) fused address simplex.
            hop_weights : (N, max_hops + 1) attention weights per hop depth.
        """
        N = x.shape[0]
        queries = self.encoder(x)  # (N, memory_dim)

        # 1. Compute Base Address Distribution A_0
        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        A0 = address_distribution(attn_scores, self.attention_fn, dim=-1)  # (N, M)

        # 2. Extract structural off-diagonal edges from base_adjacency
        coo = base_adjacency.coalesce()
        indices = coo.indices()
        values = coo.values()
        mask_off = indices[0] != indices[1]
        off_row = indices[0][mask_off]
        off_col = indices[1][mask_off]
        off_vals = values[mask_off]

        # 3. Dynamic Boundary-Gated Propagation Loop (1 ... max_hops)
        address_scales = [A0]
        current_A = A0

        if off_row.numel() > 0:
            self_indices = torch.arange(N, device=x.device, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            self_vals = torch.ones(N, device=x.device, dtype=x.dtype)

            for _ in range(self.max_hops):
                # Address L2 squared distance across structural edges
                A_row = current_A[off_row]
                A_col = current_A[off_col]
                dist_sq = ((A_row - A_col) ** 2).sum(dim=-1)

                # Edge gate G = exp(-dist_sq)
                G = torch.exp(-dist_sq)
                gated_vals = off_vals * G

                # Combine gated structural edges + full self-loops
                comb_indices = torch.cat([torch.stack([off_row, off_col]), self_indices], dim=1)
                comb_vals = torch.cat([gated_vals, self_vals], dim=0)

                # Row-normalize dynamically
                deg = torch.zeros(N, device=x.device, dtype=x.dtype).scatter_add(0, comb_indices[0], comb_vals)
                inv_deg = 1.0 / deg.clamp_min(1e-12)
                norm_vals = comb_vals * inv_deg[comb_indices[0]]

                # Build row-stochastic sparse propagation matrix for step k
                adj_k = torch.sparse_coo_tensor(comb_indices, norm_vals, (N, N)).coalesce()

                # Propagate address simplex
                current_A = torch.sparse.mm(adj_k, current_A)
                address_scales.append(current_A)
        else:
            for _ in range(self.max_hops):
                address_scales.append(current_A)

        # 4. Hop Attention Pooling
        A_stack = torch.stack(address_scales, dim=1)  # (N, max_hops + 1, M)
        hop_logits = self.hop_attention(queries)      # (N, max_hops + 1)
        hop_weights = F.softmax(hop_logits, dim=-1)   # (N, max_hops + 1)
        self.last_hop_weights = hop_weights

        # Fused address distribution
        A_fused = (hop_weights.unsqueeze(-1) * A_stack).sum(dim=1)  # (N, M)

        # Read out final embedding
        embedding = torch.matmul(A_fused, self.memory_values)  # (N, memory_dim)
        return embedding, A_fused, hop_weights


class BAAPMemoryAutoencoder(nn.Module):
    """BAAPMemoryLayer + decoder back to feature space."""

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
        self.memory = BAAPMemoryLayer(
            feature_dim=feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            max_hops=max_hops,
            temperature=temperature,
            attention_fn=attention_fn,
        )
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x, base_adjacency):
        embedding, attn_weights, hop_weights = self.memory(x, base_adjacency)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, attn_weights, hop_weights
