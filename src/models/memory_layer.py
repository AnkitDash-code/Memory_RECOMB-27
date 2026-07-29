import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddedMemoryLayer(nn.Module):
    def __init__(self, feature_dim, memory_slots=512, memory_dim=128):
        super().__init__()
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, feature_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.query_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        queries = self.query_proj(x)
        attn_scores = torch.matmul(queries, self.memory_keys.T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, self.memory_values), attn_weights


def attention_entropy(attn_weights):
    """Per-row entropy of the attention distribution, in nats."""
    eps = 1e-12
    return -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)


class EmbeddedMemoryAutoencoder(nn.Module):
    """Trainable wrapper around EmbeddedMemoryLayer.

    EmbeddedMemoryLayer alone has no loss signal: nothing pushes memory_keys
    or memory_values away from their random init. This adds a linear decoder
    back to feature space so the layer can be trained with a reconstruction
    objective, exposing the underlying memory-addressing mechanism to actual
    gradient signal instead of random output.
    """

    def __init__(self, feature_dim, memory_slots=512, memory_dim=128):
        super().__init__()
        self.memory = EmbeddedMemoryLayer(feature_dim, memory_slots, memory_dim)
        self.decoder = nn.Linear(memory_dim, feature_dim)

    def forward(self, x):
        embedding, attn_weights = self.memory(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, attn_weights


def spatial_smoothness_loss(embedding, edge_index, edge_weight=None):
    """Mean squared distance between embeddings of spatially-connected spots.

    edge_index: (2, n_edges) long tensor of (row, col) indices from the
    spatial connectivity graph. This is the paper's "memory-addressing
    replacing message passing" mechanism in the loss: instead of an explicit
    GNN layer propagating features along the spatial graph, neighboring spots
    are only encouraged (via this penalty) to land on similar memory
    addresses, with all cross-spot mixing happening through the shared
    memory bank rather than direct neighbor aggregation.
    """
    row, col = edge_index
    diff = embedding[row] - embedding[col]
    sq_dist = (diff**2).sum(dim=-1)
    if edge_weight is not None:
        sq_dist = sq_dist * edge_weight
    return sq_dist.mean()


def connectivities_to_edge_index(connectivities):
    """Convert an adata.obsp['spatial_connectivities'] sparse matrix to
    (edge_index, edge_weight) torch tensors."""
    coo = connectivities.tocoo()
    edge_index = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float32)
    return edge_index, edge_weight


def main():
    import math

    from src.data.load_visium import load_visium_crop
    from src.data.preprocess import get_pca_features, preprocess

    adata = preprocess(load_visium_crop())
    x = torch.tensor(get_pca_features(adata), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = EmbeddedMemoryLayer(feature_dim=x.shape[1]).to(device)
    x = x.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _, attn_weights = layer(x)

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"peak VRAM: {peak_mb:.1f} MB")
    else:
        print("running on CPU; no VRAM to report")

    entropy = attention_entropy(attn_weights)
    max_entropy = math.log(layer.memory_keys.shape[0])
    print(f"attn_weights shape: {tuple(attn_weights.shape)}")
    print(f"median entropy: {entropy.median().item():.4f} (max possible: {max_entropy:.4f})")


if __name__ == "__main__":
    main()
