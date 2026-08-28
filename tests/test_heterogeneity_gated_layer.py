import numpy as np
import scipy.sparse as sp
import torch

from src.models.heterogeneity_gated_layer import (
    HeterogeneityGatedMemoryAutoencoder,
    HeterogeneityGatedMemoryLayer,
    compute_fixed_certainty,
    heterogeneity_gated_propagation,
)
from src.models.memory_layer import normalized_adjacency


def _ring(n):
    rows = list(range(n)) + list(range(n))
    cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# compute_fixed_certainty: must be a pure, deterministic function of
# precomputed data, never of any trainable parameter.
# ---------------------------------------------------------------------------


def test_compute_fixed_certainty_returns_numpy_not_torch():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 8)).astype(np.float32)
    conn = _ring(20)
    certainty = compute_fixed_certainty(features, conn, physical_radius_um=110.0, platform="visium", avg_edge_length_um=55.0)
    assert isinstance(certainty, np.ndarray)
    assert certainty.dtype == np.float32


def test_compute_fixed_certainty_shape_and_range():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(30, 6)).astype(np.float32)
    conn = _ring(30)
    certainty = compute_fixed_certainty(features, conn, physical_radius_um=165.0, platform="visium", avg_edge_length_um=55.0)
    assert certainty.shape == (30,)
    assert np.all(certainty >= 0.0) and np.all(certainty <= 1.0)
    assert not np.any(np.isnan(certainty))


def test_compute_fixed_certainty_is_deterministic():
    """Calling twice on identical inputs must give identical output -- this is
    the core invariant the whole design depends on: nothing stochastic or
    trainable is involved.
    """
    rng = np.random.default_rng(2)
    features = rng.normal(size=(25, 5)).astype(np.float32)
    conn = _ring(25)
    c1 = compute_fixed_certainty(features, conn, physical_radius_um=220.0, platform="visium", avg_edge_length_um=55.0)
    c2 = compute_fixed_certainty(features, conn, physical_radius_um=220.0, platform="visium", avg_edge_length_um=55.0)
    assert np.array_equal(c1, c2)


def test_compute_fixed_certainty_is_rank_based_not_scale_dependent():
    """Scaling all features by a large constant must not change the certainty
    ranking (rank-based percentile, not raw min-max) -- this is what makes the
    statistic comparable across datasets with different absolute expression-
    distance magnitudes.
    """
    rng = np.random.default_rng(3)
    features = rng.normal(size=(20, 4)).astype(np.float32)
    conn = _ring(20)
    c_small = compute_fixed_certainty(features, conn, physical_radius_um=110.0, platform="visium", avg_edge_length_um=55.0)
    c_scaled = compute_fixed_certainty(features * 1000.0, conn, physical_radius_um=110.0, platform="visium", avg_edge_length_um=55.0)
    assert np.allclose(c_small, c_scaled, atol=1e-4)


# ---------------------------------------------------------------------------
# heterogeneity_gated_propagation: math invariants + the no-gradient-to-
# certainty guarantee.
# ---------------------------------------------------------------------------


def test_heterogeneity_gated_propagation_zero_certainty_stays_unpropagated():
    n, k = 10, 4
    a0 = torch.softmax(torch.randn(n, k), dim=-1)
    adjacency = normalized_adjacency(_ring(n))
    certainty = torch.zeros(n)
    out = heterogeneity_gated_propagation(a0, adjacency, n_hops=3, certainty=certainty)
    assert torch.allclose(out, a0, atol=1e-5)


def test_heterogeneity_gated_propagation_full_certainty_fully_propagates():
    n, k = 10, 4
    a0 = torch.softmax(torch.randn(n, k), dim=-1)
    adjacency = normalized_adjacency(_ring(n))
    certainty = torch.ones(n)
    out = heterogeneity_gated_propagation(a0, adjacency, n_hops=3, certainty=certainty)
    expected = a0
    for _ in range(3):
        expected = torch.sparse.mm(adjacency, expected)
    assert torch.allclose(out, expected, atol=1e-5)


def test_heterogeneity_gated_propagation_output_is_valid_simplex():
    torch.manual_seed(0)
    n, k = 15, 5
    a0 = torch.softmax(torch.randn(n, k), dim=-1)
    adjacency = normalized_adjacency(_ring(n))
    certainty = torch.rand(n)
    out = heterogeneity_gated_propagation(a0, adjacency, n_hops=4, certainty=certainty)
    assert torch.allclose(out.sum(dim=-1), torch.ones(n), atol=1e-5)
    assert torch.all(out >= -1e-6)


def test_heterogeneity_gated_propagation_never_backprops_into_certainty():
    """Even if a caller mistakenly passes a certainty tensor derived from a
    trainable parameter, no gradient may reach it -- this is the hard
    guarantee the whole design rests on. Regression test for exactly the
    failure mode that made every prior learned gate collapse.
    """
    n, k = 8, 3
    a0_logits = torch.nn.Parameter(torch.randn(n, k))
    a0 = torch.softmax(a0_logits, dim=-1)  # a0 DOES require grad, so backward has something real to do
    adjacency = normalized_adjacency(_ring(n))

    bad_param = torch.nn.Parameter(torch.full((n,), 0.5))
    certainty = torch.sigmoid(bad_param)  # deliberately trainable-derived, to test the guard

    out = heterogeneity_gated_propagation(a0, adjacency, n_hops=2, certainty=certainty)
    loss = out.sum()
    loss.backward()
    assert a0_logits.grad is not None  # sanity: the real gradient path does work
    assert bad_param.grad is None  # detach() inside the function must have cut the certainty graph


# ---------------------------------------------------------------------------
# Full layer / autoencoder integration
# ---------------------------------------------------------------------------


def test_layer_forward_shapes_and_valid_simplex():
    torch.manual_seed(0)
    layer = HeterogeneityGatedMemoryLayer(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=2)
    x = torch.randn(12, 6)
    adjacency = normalized_adjacency(_ring(12))
    certainty = torch.rand(12)
    embedding, addresses = layer(x, adjacency, certainty)
    assert embedding.shape == (12, 4)
    assert addresses.shape == (12, 5)
    assert torch.allclose(addresses.sum(dim=-1), torch.ones(12), atol=1e-5)


def test_autoencoder_trains_gradients_flow_to_model_not_to_certainty():
    torch.manual_seed(0)
    model = HeterogeneityGatedMemoryAutoencoder(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=2)
    x = torch.randn(10, 6)
    adjacency = normalized_adjacency(_ring(10))
    certainty = torch.rand(10)  # not a parameter -- no .grad attribute to check, just confirm training works

    reconstruction, embedding, addresses = model(x, adjacency, certainty)
    assert reconstruction.shape == x.shape
    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()
    assert model.memory.memory_keys.grad is not None
    assert torch.any(model.memory.memory_keys.grad != 0)
    for p in model.memory.encoder.parameters():
        assert p.grad is not None


def test_no_nans_through_full_forward_and_backward():
    torch.manual_seed(0)
    model = HeterogeneityGatedMemoryAutoencoder(feature_dim=8, memory_slots=6, memory_dim=5, hidden_dim=10, n_hops=3)
    x = torch.randn(20, 8)
    adjacency = normalized_adjacency(_ring(20))
    certainty = torch.rand(20)
    reconstruction, embedding, addresses = model(x, adjacency, certainty)
    assert not torch.any(torch.isnan(reconstruction))
    assert not torch.any(torch.isnan(embedding))
    assert not torch.any(torch.isnan(addresses))
    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert not torch.any(torch.isnan(p.grad))
