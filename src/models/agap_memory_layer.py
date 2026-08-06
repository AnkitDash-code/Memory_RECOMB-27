"""Attention-Guided Address Propagation (AGAP).

AGAP builds a dynamic sparse adjacency from latent query similarity rather than
using the static spatial connectivity weights directly. The intent is to open
propagation inside homogeneous domains while reducing cross-boundary mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _segment_softmax(scores, segment_ids, num_segments):
    """Softmax over values grouped by segment id.

    This is a small dependency-free fallback for environments without
    torch_scatter. The datasets in this repo are small enough that the looped
    fallback is still practical when needed.
    """
    if scores.numel() == 0:
        return scores

    if hasattr(torch.Tensor, "scatter_reduce_"):
        max_values = torch.full((num_segments,), -torch.inf, device=scores.device, dtype=scores.dtype)
        max_values.scatter_reduce_(0, segment_ids, scores, reduce="amax", include_self=True)
        stabilized = scores - max_values[segment_ids]
        exp_scores = torch.exp(stabilized)
        denom = torch.zeros(num_segments, device=scores.device, dtype=scores.dtype)
        denom.scatter_add_(0, segment_ids, exp_scores)
        return exp_scores / (denom[segment_ids] + 1e-12)

    weights = torch.zeros_like(scores)
    for seg in torch.unique(segment_ids):
        mask = segment_ids == seg
        weights[mask] = torch.softmax(scores[mask], dim=0)
    return weights


def _row_normalize_sparse(indices, values, size, device, dtype):
    row = indices[0]
    degree = torch.zeros(size[0], device=device, dtype=dtype)
    degree.scatter_add_(0, row, values)
    normalized = values / (degree[row] + 1e-12)
    return torch.sparse_coo_tensor(indices, normalized, size).coalesce()


class AGAPMemoryAutoencoder(nn.Module):
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

    def forward(self, x, edge_index):
        queries = self.encoder(x)

        base_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        a_0 = F.softmax(base_scores, dim=-1)

        n_spots = x.shape[0]
        device = x.device
        dtype = queries.dtype

        if edge_index is None or edge_index.numel() == 0:
            current_a = a_0
        else:
            src, dst = edge_index
            q_src = queries[src]
            q_dst = queries[dst]
            scores = (q_src * q_dst).sum(dim=-1) / self.temperature
            alpha = _segment_softmax(scores, dst, n_spots)

            self_loop_index = torch.arange(n_spots, device=device, dtype=torch.long)
            all_rows = torch.cat([dst, self_loop_index], dim=0)
            all_cols = torch.cat([src, self_loop_index], dim=0)
            all_weights = torch.cat([
                alpha,
                torch.ones(n_spots, device=device, dtype=dtype),
            ], dim=0)
            adj_index = torch.stack([all_rows, all_cols], dim=0)
            adj_dynamic = _row_normalize_sparse(
                adj_index,
                all_weights,
                (n_spots, n_spots),
                device=device,
                dtype=dtype,
            )

            current_a = a_0
            for _ in range(self.n_hops):
                current_a = torch.sparse.mm(adj_dynamic, current_a)

        embedding = torch.matmul(current_a, self.memory_values)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, current_a