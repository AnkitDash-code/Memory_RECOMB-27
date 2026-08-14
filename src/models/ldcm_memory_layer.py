"""Latent Denoising & Contrastive Memory (LDCM).

LDCM keeps the stable address-propagation baseline, but smooths the latent
queries before the memory lookup and adds an embedding-level projection head
for contrastive regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LDCMMemoryAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.n_hops = n_hops
        self.temperature = temperature

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
        self.projection_head = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )

    def forward(self, x, adjacency):
        q_initial = self.encoder(x)
        if adjacency is not None:
            q_smoothed = torch.sparse.mm(adjacency, q_initial)
            queries = q_initial + q_smoothed
        else:
            queries = q_initial

        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        a_0 = F.softmax(attn_scores, dim=-1)

        a = a_0
        if adjacency is not None:
            for _ in range(self.n_hops):
                a = torch.sparse.mm(adjacency, a)

        embedding = torch.matmul(a, self.memory_values)
        reconstruction = self.decoder(embedding)
        projection = self.projection_head(embedding)
        return reconstruction, embedding, projection, a