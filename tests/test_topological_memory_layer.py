import numpy as np
import scipy.sparse as sp
import torch

from src.models.memory_layer import normalized_adjacency
from src.models.topological_memory_layer import (
    TopologicalMemoryAutoencoder,
    TopologicalMemoryLayer,
)


def _ring_connectivities(n):
    rows = list(range(n)) + list(range(n))
    cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_som_kernel_is_symmetric_with_unit_diagonal():
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8)
    kernel = layer._som_neighbor_kernel()

    assert kernel.shape == (8, 8)
    assert torch.allclose(torch.diagonal(kernel), torch.ones(8))
    assert torch.allclose(kernel, kernel.T)


def test_som_kernel_decays_with_slot_distance():
    """The whole point of the topology: slot 0 must be more strongly coupled
    to slot 1 than to slot 7."""
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8, som_sigma=1.5)
    kernel = layer._som_neighbor_kernel()

    assert kernel[0, 1] > kernel[0, 3] > kernel[0, 7]


def test_som_kernel_is_not_degenerately_flat_at_default_sigma():
    """Regression test for a real bug in the source plan: with slot positions
    on a linspace(0, 1) scale, the default som_sigma=1.5 makes every kernel
    entry >= 0.8 -- an almost flat kernel that pulls every slot toward every
    input, silently turning the mechanism into a no-op. Integer slot indices
    keep the kernel meaningfully peaked."""
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=16, memory_dim=8, som_sigma=1.5)
    kernel = layer._som_neighbor_kernel()

    # Furthest pair must be strongly decoupled, not ~0.8 as the linspace version gives.
    assert kernel[0, -1] < 0.01


def test_larger_sigma_flattens_kernel():
    narrow = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8, som_sigma=0.5)
    wide = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8, som_sigma=2.5)

    assert wide._som_neighbor_kernel()[0, 4] > narrow._som_neighbor_kernel()[0, 4]


def test_expected_position_is_within_unit_interval():
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8)
    attn = torch.softmax(torch.randn(20, 8), dim=-1)

    pos = layer.expected_position(attn)

    assert pos.shape == (20,)
    assert (pos >= 0).all() and (pos <= 1).all()


def test_expected_position_tracks_which_slot_is_addressed():
    """A spot committed to slot 0 should sit at position 0; one committed to
    the last slot at position 1."""
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8)
    attn = torch.zeros(2, 8)
    attn[0, 0] = 1.0
    attn[1, 7] = 1.0

    pos = layer.expected_position(attn)

    assert pos[0].item() == 0.0
    assert pos[1].item() == 1.0


def test_ordinal_smoothness_loss_zero_when_all_positions_equal():
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8)
    adjacency = normalized_adjacency(_ring_connectivities(10))
    expected_pos = torch.full((10,), 0.5)

    loss = layer.ordinal_smoothness_loss(expected_pos, adjacency)

    assert loss.item() < 1e-8


def test_ordinal_smoothness_loss_penalizes_neighbor_jumps():
    """Alternating positions around a ring (every neighbor maximally far in
    ordinal space) must score higher than a smooth ramp."""
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8)
    adjacency = normalized_adjacency(_ring_connectivities(10))

    alternating = torch.tensor([0.0, 1.0] * 5)
    smooth = torch.linspace(0, 1, 10)

    assert (
        layer.ordinal_smoothness_loss(alternating, adjacency).item()
        > layer.ordinal_smoothness_loss(smooth, adjacency).item()
    )


def test_som_topology_loss_is_positive_and_differentiable():
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8)
    queries = torch.randn(12, 8)
    attn = torch.softmax(torch.randn(12, 8), dim=-1)

    loss = layer.som_topology_loss(queries, attn)
    loss.backward()

    assert loss.item() > 0
    assert layer.memory_keys.grad is not None
    assert not torch.isnan(layer.memory_keys.grad).any()


def test_som_topology_loss_lower_when_keys_match_their_winners():
    """Sanity check that the loss actually measures what it claims: keys
    placed on top of the data should score lower than keys far from it."""
    torch.manual_seed(0)
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=4, memory_dim=3)
    queries = torch.zeros(8, 3)
    attn = torch.softmax(torch.randn(8, 4), dim=-1)

    with torch.no_grad():
        layer.memory_keys.copy_(torch.zeros(4, 3))
    close = layer.som_topology_loss(queries, attn).item()

    with torch.no_grad():
        layer.memory_keys.copy_(torch.full((4, 3), 10.0))
    far = layer.som_topology_loss(queries, attn).item()

    assert far > close


def test_forward_returns_internals_with_expected_shapes():
    n_spots, feature_dim, n_slots, memory_dim = 10, 16, 8, 4
    layer = TopologicalMemoryLayer(
        feature_dim, memory_slots=n_slots, memory_dim=memory_dim, n_hops=2
    )
    x = torch.randn(n_spots, feature_dim)
    adjacency = normalized_adjacency(_ring_connectivities(n_spots))

    embedding, propagated, pre_attn, queries = layer(x, adjacency)

    assert embedding.shape == (n_spots, memory_dim)
    assert propagated.shape == (n_spots, n_slots)
    assert pre_attn.shape == (n_spots, n_slots)
    assert queries.shape == (n_spots, memory_dim)
    # Propagated addresses must remain a valid simplex.
    assert torch.allclose(propagated.sum(dim=-1), torch.ones(n_spots), atol=1e-5)


def test_propagation_smooths_expected_position():
    """Address propagation should make neighboring spots' ordinal positions
    more similar -- confirming the topology interacts with the existing
    spatial mechanism rather than being independent of it."""
    torch.manual_seed(0)
    n_spots = 30
    layer = TopologicalMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8, n_hops=4)
    x = torch.randn(n_spots, 16)
    adjacency = normalized_adjacency(_ring_connectivities(n_spots))

    _, propagated, pre_attn, _ = layer(x, adjacency)
    pos_before = layer.expected_position(pre_attn)
    pos_after = layer.expected_position(propagated)

    assert (
        layer.ordinal_smoothness_loss(pos_after, adjacency).item()
        < layer.ordinal_smoothness_loss(pos_before, adjacency).item()
    )


def test_autoencoder_reconstructs_feature_dim():
    n_spots, feature_dim = 12, 20
    model = TopologicalMemoryAutoencoder(feature_dim, memory_slots=8, memory_dim=8)
    x = torch.randn(n_spots, feature_dim)
    adjacency = normalized_adjacency(_ring_connectivities(n_spots))

    reconstruction, embedding, propagated, pre_attn, queries = model(x, adjacency)

    assert reconstruction.shape == x.shape
    assert embedding.shape == (n_spots, 8)
