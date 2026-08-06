"""Personalized PageRank (PPR) address propagation.

PPR anchors each propagation step back to the original 0-hop address so deep
propagation can denoise without fully overwriting spot identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPRMemoryAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
        alpha=0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.n_hops = n_hops
        self.temperature = temperature
        self.alpha = alpha

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

        self.memory_keys = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)

        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x, adjacency):
        queries = self.encoder(x)
        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        a_0 = F.softmax(attn_scores, dim=-1)

        propagated = a_0
        if adjacency is not None:
            for _ in range(self.n_hops):
                propagated = (1.0 - self.alpha) * torch.sparse.mm(adjacency, propagated) + self.alpha * a_0

        embedding = torch.matmul(propagated, self.memory_values)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, propagated