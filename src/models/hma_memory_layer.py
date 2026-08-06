"""Hopfield Memory Attractor (HMA) -- novel architecture for RECOMB 2027.

Hypothesis: Deep spatial propagation (e.g. n_hops=4) is essential for denoising
sparse, dropout-heavy spatial transcriptomics features. However, deep propagation
blurs address distributions along domain boundaries.

HMA introduces Modern Hopfield Attractor Dynamics following deep propagation:
1. Base query-key address computation: A_0 = softmax( Q @ K^T / temperature )
2. Deep propagation for n_hops: A_blurred = (D^-1 (A+I))^n_hops A_0
3. Hopfield Attractor Update:
   - Project blurred address back into key space: Z_blurred = A_blurred @ K
   - Re-query memory key bank K: S = Z_blurred @ K^T = A_blurred @ (K @ K^T)
   - Apply attractor inverse-temperature scaling: A_sharp = softmax( beta * S )
   where beta is a learnable scalar parameter.

The attractor update "snaps" mixed boundary addresses back to clean memory
prototypes while preserving the deep denoising achieved during propagation.

No pre-existing files are modified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.memory_layer import address_distribution


class HMAMemoryLayer(nn.Module):
    """Hopfield Memory Attractor Layer.

    Parameters
    ----------
    feature_dim : int
        Input feature dimension (e.g. 3000 HVGs).
    memory_slots : int
        Number of memory slots M (default 16).
    memory_dim : int
        Dimension of key and value vectors d (default 128).
    hidden_dim : int
        Hidden dimension for encoder (default 256).
    n_hops : int
        Number of spatial address propagation hops (default 4).
    temperature : float
        Query-key softmax scaling factor (default 1.0).
    attention_fn : str
        Normalization function for address distribution ("softmax" default).
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
        self.feature_dim = feature_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.n_hops = n_hops
        self.temperature = temperature
        self.attention_fn = attention_fn

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        self.memory_keys = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)

        # Learnable attractor inverse temperature (beta) initialized to 1.0
        self.attractor_beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, adjacency=None):
        queries = self.encoder(x)  # (N, memory_dim)

        # 1. Base Query-Key Address Distribution A_0
        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        A0 = address_distribution(attn_scores, self.attention_fn, dim=-1)  # (N, M)

        # 2. Deep Address Propagation (n_hops)
        A_blurred = A0
        if adjacency is not None:
            for _ in range(self.n_hops):
                A_blurred = torch.sparse.mm(adjacency, A_blurred)

        # 3. Hopfield Attractor Dynamics Update
        # Project blurred address back into key space: Z_blurred = A_blurred @ K
        Z_blurred = torch.matmul(A_blurred, self.memory_keys)  # (N, memory_dim)

        # Compute attractor energy scores: S = Z_blurred @ K^T
        attractor_scores = torch.matmul(Z_blurred, self.memory_keys.T)  # (N, M)

        # Snap mixed addresses to biological prototypes using learnable beta
        A_sharp = address_distribution(self.attractor_beta * attractor_scores, self.attention_fn, dim=-1)  # (N, M)

        # 4. Final Readout Embedding
        embedding = torch.matmul(A_sharp, self.memory_values)  # (N, memory_dim)
        return embedding, A_sharp


class HopfieldMemoryAutoencoder(nn.Module):
    """HMAMemoryLayer + feature reconstruction decoder."""

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
        self.memory = HMAMemoryLayer(
            feature_dim=feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
            attention_fn=attention_fn,
        )
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x, adjacency=None):
        embedding, attn_weights = self.memory(x, adjacency)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, attn_weights
