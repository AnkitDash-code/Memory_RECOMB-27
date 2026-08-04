import numpy as np
import scipy.sparse as sp
import torch

from src.models.memory_layer import (
    HopFusionMemoryAutoencoder,
    HopFusionMemoryLayer,
    address_spatial_coherence_loss,
    normalized_adjacency,
)
from src.models.train_hop_fusion import mean_address_entropy_by_hop


def _ring(n):
    rows = list(range(n)) + list(range(n))
    cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_hop_fusion_concatenates_all_depths_and_heterogeneity():
    torch.manual_seed(0)
    layer = HopFusionMemoryLayer(
        feature_dim=6,
        memory_slots=4,
        memory_dim=5,
        hidden_dim=8,
        max_hops=2,
        fusion_hidden_dim=7,
        fusion_depth=2,
    )
    x = torch.randn(10, 6)
    score = torch.linspace(0, 1, 10)
    embedding, addresses = layer(x, normalized_adjacency(_ring(10)), score)

    assert embedding.shape == (10, 5)
    assert addresses.shape == (10, 4)
    assert layer.last_fusion_input.shape == (10, 3 * 4 + 1)
    assert torch.allclose(layer.last_fusion_input[:, -1], score)
    assert len(layer.last_address_by_hop) == 3
    assert all(torch.allclose(view.sum(dim=-1), torch.ones(10)) for view in layer.last_address_by_hop)
    assert not hasattr(layer, "hop_gate")


def test_hop_fusion_autoencoder_trains_and_returns_expected_shapes():
    torch.manual_seed(1)
    model = HopFusionMemoryAutoencoder(
        feature_dim=5,
        memory_slots=4,
        memory_dim=3,
        hidden_dim=7,
        max_hops=1,
        fusion_hidden_dim=6,
    )
    x = torch.randn(8, 5)
    adjacency = normalized_adjacency(_ring(8))
    reconstruction, embedding, addresses = model(
        x, adjacency, torch.zeros(8)
    )
    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()
    assert reconstruction.shape == x.shape
    assert embedding.shape == (8, 3)
    assert addresses.shape == (8, 4)
    assert model.memory.fusion_mlp[0].weight.grad is not None


def test_address_coherence_loss_is_zero_for_identical_addresses():
    addresses = torch.ones(4, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_weight = torch.ones(3)
    assert address_spatial_coherence_loss(addresses, edge_index, edge_weight).item() == 0.0


def test_mean_address_entropy_by_hop_is_rowwise_and_maskable():
    one_hot = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    uniform = torch.full((2, 2), 0.5)
    entropies = mean_address_entropy_by_hop([one_hot, uniform])
    assert torch.allclose(
        entropies,
        torch.tensor([0.0, np.log(2.0)], dtype=entropies.dtype),
        atol=1e-6,
    )
    masked = mean_address_entropy_by_hop([one_hot, uniform], torch.tensor([True, False]))
    assert torch.allclose(masked, entropies, atol=1e-6)
