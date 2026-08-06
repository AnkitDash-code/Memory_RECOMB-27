"""Gated Multi-Scale Memory (GMSM) architecture.

This branch keeps a local 0-hop address stream and a globally propagated
4-hop address stream separate, then learns a per-spot gate over the resulting
embeddings. The intent is to preserve sharp local structure while still
allowing deep denoising when the data support it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiScaleMemoryLayer(nn.Module):
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

        self.memory_keys_local = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_values_local = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_keys_global = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_values_global = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)

        self.gate_network = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, adjacency):
        queries = self.encoder(x)

        local_scores = torch.matmul(queries, self.memory_keys_local.T) / self.temperature
        a_local = F.softmax(local_scores, dim=-1)

        a_global = a_local
        if adjacency is not None:
            for _ in range(self.n_hops):
                a_global = torch.sparse.mm(adjacency, a_global)

        z_local = torch.matmul(a_local, self.memory_values_local)
        z_global = torch.matmul(a_global, self.memory_values_global)

        gate = torch.sigmoid(self.gate_network(queries))
        z_fused = gate * z_local + (1.0 - gate) * z_global

        return z_fused, a_local, a_global, gate


class GatedMultiScaleMemoryAutoencoder(nn.Module):
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
        self.memory = GatedMultiScaleMemoryLayer(
            feature_dim=feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
        )
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x, adjacency):
        embedding, a_local, a_global, gate = self.memory(x, adjacency)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, a_local, a_global, gate