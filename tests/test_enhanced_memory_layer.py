import numpy as np
import scipy.sparse as sp
import torch

from src.models.enhanced_memory_layer import (
    TwoStreamMemoryAutoencoder,
    TwoStreamMemoryLayer,
    augment_adjacency,
    augment_features,
    entropy_gated_propagation,
    key_repulsion_loss,
    kl_contrastive_address_loss,
)
from src.models.memory_layer import normalized_adjacency


def _ring(n):
    rows = list(range(n)) + list(range(n))
    cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Fix 1 -- key repulsion
# ---------------------------------------------------------------------------


def test_key_repulsion_loss_maximal_for_identical_keys():
    keys = torch.ones(5, 8)  # every row identical -> cosine sim 1 everywhere
    loss = key_repulsion_loss(keys)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)


def test_key_repulsion_loss_near_zero_for_orthogonal_keys():
    keys = torch.eye(6)  # one-hot rows are pairwise orthogonal
    loss = key_repulsion_loss(keys)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_key_repulsion_loss_gradient_pushes_keys_apart():
    torch.manual_seed(0)
    keys = torch.nn.Parameter(torch.randn(4, 3) * 0.01 + 1.0)  # start near-collapsed
    opt = torch.optim.SGD([keys], lr=0.5)
    initial = key_repulsion_loss(keys.detach()).item()
    for _ in range(50):
        opt.zero_grad()
        loss = key_repulsion_loss(keys)
        loss.backward()
        opt.step()
    final = key_repulsion_loss(keys.detach()).item()
    assert final < initial  # descending this loss must reduce mean pairwise similarity


# ---------------------------------------------------------------------------
# Fix 2 -- KL contrastive address regularisation
# ---------------------------------------------------------------------------


def test_kl_contrastive_loss_zero_for_identical_distributions():
    a = torch.softmax(torch.randn(10, 6), dim=-1)
    loss = kl_contrastive_address_loss(a, a.clone())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_kl_contrastive_loss_positive_and_symmetric_for_divergent_distributions():
    torch.manual_seed(0)
    a = torch.softmax(torch.randn(10, 6), dim=-1)
    b = torch.softmax(torch.randn(10, 6), dim=-1)
    loss_ab = kl_contrastive_address_loss(a, b)
    loss_ba = kl_contrastive_address_loss(b, a)
    assert loss_ab.item() > 0
    assert torch.isclose(loss_ab, loss_ba, atol=1e-5)


def test_augment_adjacency_preserves_row_stochastic_and_self_loops():
    adjacency = normalized_adjacency(_ring(20))
    for drop_rate in (0.0, 0.1, 0.5, 0.9):
        aug = augment_adjacency(adjacency, drop_rate=drop_rate)
        row_sums = torch.sparse.sum(aug, dim=1).to_dense()
        assert torch.allclose(row_sums, torch.ones(20), atol=1e-5)
        dense = aug.to_dense()
        assert torch.all(torch.diagonal(dense) > 0)  # self-loops never dropped


def test_augment_adjacency_zero_drop_rate_is_identity():
    adjacency = normalized_adjacency(_ring(10))
    aug = augment_adjacency(adjacency, drop_rate=0.0)
    assert torch.allclose(aug.to_dense(), adjacency.to_dense(), atol=1e-5)


def test_augment_features_mask_rate_zero_is_identity():
    torch.manual_seed(0)
    x = torch.randn(8, 5)
    assert torch.allclose(augment_features(x, mask_rate=0.0), x)


def test_augment_features_mask_rate_one_is_all_zero():
    torch.manual_seed(0)
    x = torch.randn(8, 5)
    assert torch.allclose(augment_features(x, mask_rate=1.0), torch.zeros_like(x))


# ---------------------------------------------------------------------------
# Fix 5 -- entropy-gated propagation
# ---------------------------------------------------------------------------


def test_entropy_gated_propagation_uniform_input_stays_unpropagated():
    # Maximum-entropy A0 -> certainty=0 -> output must equal A0, not A_nhops
    n, k = 10, 4
    a0 = torch.full((n, k), 1.0 / k)
    adjacency = normalized_adjacency(_ring(n))
    out = entropy_gated_propagation(a0, adjacency, n_hops=3)
    assert torch.allclose(out, a0, atol=1e-5)


def test_entropy_gated_propagation_confident_input_fully_propagates():
    # Zero-entropy (one-hot) A0 -> certainty=1 -> output must equal full n-hop propagation
    n, k = 10, 4
    a0 = torch.zeros(n, k)
    a0[:, 0] = 1.0
    adjacency = normalized_adjacency(_ring(n))
    out = entropy_gated_propagation(a0, adjacency, n_hops=3)

    expected = a0
    for _ in range(3):
        expected = torch.sparse.mm(adjacency, expected)
    assert torch.allclose(out, expected, atol=1e-5)


def test_entropy_gated_propagation_output_is_valid_simplex():
    torch.manual_seed(0)
    n, k = 12, 5
    logits = torch.randn(n, k)
    a0 = torch.softmax(logits, dim=-1)
    adjacency = normalized_adjacency(_ring(n))
    out = entropy_gated_propagation(a0, adjacency, n_hops=4)
    assert torch.allclose(out.sum(dim=-1), torch.ones(n), atol=1e-5)
    assert torch.all(out >= -1e-6)


# ---------------------------------------------------------------------------
# Fix 4 -- two-stream memory layer
# ---------------------------------------------------------------------------


def test_two_stream_layer_shapes_and_valid_simplices():
    torch.manual_seed(0)
    layer = TwoStreamMemoryLayer(
        feature_dim=6, n_domain_slots=4, n_state_slots=3, memory_dim=5, hidden_dim=8, n_hops=2
    )
    x = torch.randn(10, 6)
    adjacency = normalized_adjacency(_ring(10))
    embedding, a_domain, a_state = layer(x, adjacency)

    assert embedding.shape == (10, 10)  # 2 * memory_dim
    assert a_domain.shape == (10, 4)
    assert a_state.shape == (10, 3)
    assert torch.allclose(a_domain.sum(dim=-1), torch.ones(10), atol=1e-5)
    assert torch.allclose(a_state.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_two_stream_layer_state_stream_is_not_propagated():
    """Regression test for the architecture's core claim: the state stream must
    be identical whether or not a spatial graph is supplied, since Fix 4's whole
    point is that state addressing stays purely local (no adjacency dependence).
    """
    torch.manual_seed(0)
    layer = TwoStreamMemoryLayer(
        feature_dim=6, n_domain_slots=4, n_state_slots=3, memory_dim=5, hidden_dim=8, n_hops=2
    )
    x = torch.randn(10, 6)
    adjacency = normalized_adjacency(_ring(10))

    _, a_domain_prop, a_state_prop = layer(x, adjacency)
    _, a_domain_noprop, a_state_noprop = layer(x, adjacency=None)

    assert torch.allclose(a_state_prop, a_state_noprop, atol=1e-6)
    assert not torch.allclose(a_domain_prop, a_domain_noprop, atol=1e-3)


def test_two_stream_layer_entropy_gate_changes_domain_addresses():
    """Sanity check the entropy_gate flag actually has an effect: with a
    non-trivial (mixed-entropy) A0, gated and ungated propagation must differ.
    """
    torch.manual_seed(3)
    kwargs = dict(feature_dim=6, n_domain_slots=4, n_state_slots=3, memory_dim=5, hidden_dim=8, n_hops=3)
    layer_gated = TwoStreamMemoryLayer(entropy_gate=True, **kwargs)
    layer_plain = TwoStreamMemoryLayer(entropy_gate=False, **kwargs)
    layer_plain.load_state_dict(layer_gated.state_dict())  # identical weights

    x = torch.randn(10, 6)
    adjacency = normalized_adjacency(_ring(10))
    _, a_domain_gated, _ = layer_gated(x, adjacency)
    _, a_domain_plain, _ = layer_plain(x, adjacency)

    assert not torch.allclose(a_domain_gated, a_domain_plain, atol=1e-4)


def test_two_stream_autoencoder_trains_and_gradients_flow_to_both_streams():
    torch.manual_seed(0)
    model = TwoStreamMemoryAutoencoder(
        feature_dim=5, n_domain_slots=3, n_state_slots=3, memory_dim=4, hidden_dim=6, n_hops=1
    )
    x = torch.randn(8, 5)
    adjacency = normalized_adjacency(_ring(8))
    reconstruction, embedding, a_domain, a_state = model(x, adjacency)

    assert reconstruction.shape == x.shape
    assert embedding.shape == (8, 8)  # 2 * memory_dim

    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()
    assert model.memory.domain_keys.grad is not None
    assert model.memory.state_keys.grad is not None
    assert torch.any(model.memory.domain_keys.grad != 0)
    assert torch.any(model.memory.state_keys.grad != 0)
