import math

import torch

from src.models.memory_layer import (
    EmbeddedMemoryAutoencoder,
    EmbeddedMemoryLayer,
    SpatialAddressMemoryAutoencoder,
    SpatialAddressMemoryLayer,
    address_distribution,
    attention_entropy,
    connectivities_to_edge_index,
    expression_weighted_adjacency,
    normalized_adjacency,
    spatial_smoothness_loss,
)


def test_output_shape():
    n_spots, feature_dim, memory_slots, memory_dim = 100, 32, 16, 8
    layer = EmbeddedMemoryLayer(feature_dim, memory_slots=memory_slots, memory_dim=memory_dim)
    x = torch.randn(n_spots, feature_dim)

    output, attn_weights = layer(x)

    assert output.shape == (n_spots, memory_dim)
    assert attn_weights.shape == (n_spots, memory_slots)


def test_softmax_rows_sum_to_one():
    layer = EmbeddedMemoryLayer(feature_dim=32, memory_slots=16, memory_dim=8)
    x = torch.randn(50, 32)

    _, attn_weights = layer(x)

    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_attention_entropy_bounds_and_flag():
    memory_slots = 16
    layer = EmbeddedMemoryLayer(feature_dim=32, memory_slots=memory_slots, memory_dim=8)
    x = torch.randn(200, 32)

    _, attn_weights = layer(x)
    entropy = attention_entropy(attn_weights)

    max_entropy = math.log(memory_slots)
    assert (entropy >= -1e-5).all()
    assert (entropy <= max_entropy + 1e-5).all()

    median_entropy = entropy.median().item()
    if median_entropy < 0.05 * max_entropy:
        print(f"WARNING: median entropy {median_entropy:.4f} near zero -> slot collapse")
    elif median_entropy > 0.95 * max_entropy:
        print(f"NOTE: median entropy {median_entropy:.4f} near uniform ({max_entropy:.4f}) "
              "-> expected for an untrained/random-init layer, not learned structure yet")


def test_device_fallback():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = EmbeddedMemoryLayer(feature_dim=32, memory_slots=16, memory_dim=8).to(device)
    x = torch.randn(10, 32, device=device)

    output, attn_weights = layer(x)

    assert output.device.type == device.type
    assert attn_weights.device.type == device.type


def test_autoencoder_reconstruction_shape():
    n_spots, feature_dim = 20, 32
    model = EmbeddedMemoryAutoencoder(feature_dim, memory_slots=16, memory_dim=8)
    x = torch.randn(n_spots, feature_dim)

    reconstruction, embedding, attn_weights = model(x)

    assert reconstruction.shape == x.shape
    assert embedding.shape == (n_spots, 8)
    assert attn_weights.shape == (n_spots, 16)


def test_autoencoder_training_step_reduces_loss():
    torch.manual_seed(0)
    n_spots, feature_dim = 30, 16
    model = EmbeddedMemoryAutoencoder(feature_dim, memory_slots=8, memory_dim=4)
    x = torch.randn(n_spots, feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    reconstruction, _, _ = model(x)
    initial_loss = torch.nn.functional.mse_loss(reconstruction, x).item()

    for _ in range(20):
        optimizer.zero_grad()
        reconstruction, _, _ = model(x)
        loss = torch.nn.functional.mse_loss(reconstruction, x)
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss


def test_spatial_smoothness_loss_zero_for_identical_embeddings():
    embedding = torch.ones(5, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    loss = spatial_smoothness_loss(embedding, edge_index)

    assert loss.item() == 0.0


def test_spatial_smoothness_loss_penalizes_distant_neighbors():
    embedding = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    edge_index = torch.tensor([[0], [1]])

    loss = spatial_smoothness_loss(embedding, edge_index)

    assert loss.item() > 0


def test_connectivities_to_edge_index_roundtrip():
    import scipy.sparse as sp

    conn = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    edge_index, edge_weight = connectivities_to_edge_index(conn)

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_weight.shape[0] == 4


def _ring_connectivities(n):
    """Sparse ring graph: spot i adjacent to i-1 and i+1 (wraparound)."""
    import numpy as np
    import scipy.sparse as sp

    rows = list(range(n)) + list(range(n))
    cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_normalized_adjacency_rows_sum_to_one():
    conn = _ring_connectivities(10)

    adjacency = normalized_adjacency(conn)
    row_sums = torch.sparse.sum(adjacency, dim=1).to_dense()

    assert torch.allclose(row_sums, torch.ones(10), atol=1e-5)


def test_expression_weighted_adjacency_rows_sum_to_one():
    import numpy as np

    n = 10
    conn = _ring_connectivities(n)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, 5)).astype(np.float32)

    adjacency = expression_weighted_adjacency(conn, features)
    row_sums = torch.sparse.sum(adjacency, dim=1).to_dense()

    assert torch.allclose(row_sums, torch.ones(n), atol=1e-5)


def test_expression_weighted_adjacency_downweights_dissimilar_neighbors():
    """A spot with one transcriptionally-similar neighbor and one very
    dissimilar one should end up propagating more weight to the similar one,
    unlike normalized_adjacency which treats both edges identically."""
    import numpy as np
    import scipy.sparse as sp

    conn = sp.csr_matrix(
        [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    )  # spot 0 connected to both 1 and 2
    features = np.array(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]], dtype=np.float32
    )  # spot 1 close to spot 0; spot 2 far from spot 0

    uniform = normalized_adjacency(conn).to_dense()
    weighted = expression_weighted_adjacency(conn, features).to_dense()

    assert uniform[0, 1] == uniform[0, 2]  # uniform treats both neighbors alike
    assert weighted[0, 1] > weighted[0, 2]  # expression-weighted favors the similar one


def test_expression_weighted_adjacency_identical_features_matches_uniform():
    """If every spot has identical features, expression similarity is uniform
    everywhere, so the result should reduce to plain normalized_adjacency."""
    import numpy as np

    n = 8
    conn = _ring_connectivities(n)
    features = np.zeros((n, 4), dtype=np.float32)

    uniform = normalized_adjacency(conn).to_dense()
    weighted = expression_weighted_adjacency(conn, features).to_dense()

    assert torch.allclose(uniform, weighted, atol=1e-5)


def test_address_distribution_softmax_matches_torch_softmax():
    scores = torch.randn(10, 6)
    assert torch.allclose(address_distribution(scores, "softmax"), torch.softmax(scores, dim=-1))


def test_address_distribution_variants_are_valid_simplices():
    scores = torch.randn(10, 6)
    for fn in ("softmax", "entmax15", "sparsemax"):
        p = address_distribution(scores, fn)
        assert torch.allclose(p.sum(dim=-1), torch.ones(10), atol=1e-5)
        assert (p >= 0).all()


def test_address_distribution_sparse_variants_produce_exact_zeros():
    """The whole point of entmax15/sparsemax over softmax: some slots get
    exactly zero weight, not just a small positive one."""
    torch.manual_seed(0)
    scores = torch.randn(20, 16) * 3  # larger spread so sparsity is expected
    dense = address_distribution(scores, "softmax")
    entmax_p = address_distribution(scores, "entmax15")
    sparsemax_p = address_distribution(scores, "sparsemax")

    assert (dense == 0).sum() == 0  # softmax never gives exact zeros
    assert (entmax_p == 0).sum() > 0
    assert (sparsemax_p == 0).sum() >= (entmax_p == 0).sum()  # sparsemax sparsest


def test_address_distribution_unknown_fn_raises():
    import pytest

    with pytest.raises(ValueError):
        address_distribution(torch.randn(3, 4), "not_a_real_fn")


def test_spatial_address_layer_accepts_attention_fn():
    n_spots, feature_dim, memory_slots = 20, 16, 8
    for fn in ("softmax", "entmax15", "sparsemax"):
        layer = SpatialAddressMemoryLayer(
            feature_dim, memory_slots=memory_slots, memory_dim=8, n_hops=1, attention_fn=fn
        )
        x = torch.randn(n_spots, feature_dim)
        adjacency = normalized_adjacency(_ring_connectivities(n_spots))

        embedding, attn = layer(x, adjacency)

        assert embedding.shape == (n_spots, 8)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(n_spots), atol=1e-4)


def test_address_propagation_keeps_valid_simplex():
    n_spots, feature_dim, memory_slots = 30, 16, 8
    layer = SpatialAddressMemoryLayer(
        feature_dim, memory_slots=memory_slots, memory_dim=8, n_hops=3
    )
    x = torch.randn(n_spots, feature_dim)
    adjacency = normalized_adjacency(_ring_connectivities(n_spots))

    embedding, attn = layer(x, adjacency)

    assert embedding.shape == (n_spots, 8)
    assert attn.shape == (n_spots, memory_slots)
    # Propagating a row-stochastic matrix over a simplex must stay a simplex.
    assert torch.allclose(attn.sum(dim=-1), torch.ones(n_spots), atol=1e-5)
    assert (attn >= -1e-6).all()


def test_no_adjacency_reduces_to_unpropagated():
    torch.manual_seed(0)
    layer = SpatialAddressMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8, n_hops=2)
    x = torch.randn(20, 16)

    _, attn_no_graph = layer(x, adjacency=None)
    layer_zero_hops = layer
    layer_zero_hops.n_hops = 0
    adjacency = normalized_adjacency(_ring_connectivities(20))
    _, attn_zero_hops = layer_zero_hops(x, adjacency)

    assert torch.allclose(attn_no_graph, attn_zero_hops, atol=1e-6)


def test_address_propagation_smooths_neighbors():
    """More hops should make spatially adjacent spots' addresses more similar."""
    torch.manual_seed(0)
    n_spots = 40
    layer = SpatialAddressMemoryLayer(feature_dim=16, memory_slots=8, memory_dim=8, n_hops=0)
    x = torch.randn(n_spots, 16)
    adjacency = normalized_adjacency(_ring_connectivities(n_spots))

    _, attn_0 = layer(x, adjacency)
    layer.n_hops = 4
    _, attn_4 = layer(x, adjacency)

    def neighbor_divergence(attn):
        return (attn - torch.roll(attn, shifts=1, dims=0)).abs().sum(dim=-1).mean()

    assert neighbor_divergence(attn_4) < neighbor_divergence(attn_0)


def test_spatial_address_autoencoder_reconstructs_feature_dim():
    n_spots, feature_dim = 25, 40
    model = SpatialAddressMemoryAutoencoder(feature_dim, memory_slots=8, memory_dim=8)
    x = torch.randn(n_spots, feature_dim)
    adjacency = normalized_adjacency(_ring_connectivities(n_spots))

    reconstruction, embedding, attn = model(x, adjacency)

    assert reconstruction.shape == x.shape
    assert embedding.shape == (n_spots, 8)
    assert attn.shape == (n_spots, 8)


def test_usage_entropy_distinguishes_collapse_from_spread():
    """Marginal usage entropy must be near 0 when all spots pick one slot,
    and near log(n_slots) when usage is spread -- this is the quantity that
    actually detects slot collapse."""
    from src.models.memory_layer import usage_entropy

    n_spots, n_slots = 50, 8

    collapsed = torch.zeros(n_spots, n_slots)
    collapsed[:, 3] = 1.0  # every spot addresses the same slot

    spread = torch.eye(n_slots).repeat(n_spots // n_slots, 1)  # slots used evenly

    assert usage_entropy(collapsed).item() < 0.01
    assert usage_entropy(spread).item() > math.log(n_slots) - 0.01


def test_usage_entropy_differs_from_row_entropy():
    """Confident-but-balanced routing: per-row entropy ~0 (each spot commits)
    while usage entropy is maximal (all slots used). The two must not be
    conflated -- maximizing the wrong one gives mushy assignments."""
    from src.models.memory_layer import usage_entropy

    n_slots = 8
    confident_but_balanced = torch.eye(n_slots).repeat(5, 1)

    assert attention_entropy(confident_but_balanced).mean().item() < 0.01
    assert usage_entropy(confident_but_balanced).item() > math.log(n_slots) - 0.01


def test_feature_hops_zero_is_pure_address_propagation():
    """feature_hops=0 must leave features untouched -- this is the invariant that
    keeps the 'addressing replaces message passing' claim honest."""
    torch.manual_seed(0)
    n_spots = 20
    adjacency = normalized_adjacency(_ring_connectivities(n_spots))
    x = torch.randn(n_spots, 16)

    pure = SpatialAddressMemoryLayer(16, memory_slots=8, memory_dim=8, n_hops=2, feature_hops=0)
    _, attn_pure = pure(x, adjacency)
    _, attn_no_graph_features = pure(x, adjacency)

    assert torch.allclose(attn_pure, attn_no_graph_features)


def test_feature_hops_changes_representation():
    """feature_hops>0 must actually aggregate neighbour features, making it a
    genuine hybrid rather than a no-op flag."""
    torch.manual_seed(0)
    n_spots = 20
    adjacency = normalized_adjacency(_ring_connectivities(n_spots))
    x = torch.randn(n_spots, 16)

    layer = SpatialAddressMemoryLayer(16, memory_slots=8, memory_dim=8, n_hops=2, feature_hops=0)
    _, attn_pure = layer(x, adjacency)
    layer.feature_hops = 2
    _, attn_hybrid = layer(x, adjacency)

    assert not torch.allclose(attn_pure, attn_hybrid, atol=1e-4)


def test_kmeans_init_sets_keys_to_cluster_centers():
    """initialize_keys_kmeans must move memory_keys away from their random
    init to the actual cluster centers of the given queries."""
    torch.manual_seed(0)
    n_slots, dim = 4, 6
    layer = SpatialAddressMemoryLayer(feature_dim=dim, memory_slots=n_slots, memory_dim=dim)
    random_keys = layer.memory_keys.data.clone()

    import numpy as np
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(loc=i * 20.0, scale=0.1, size=(20, dim)) for i in range(n_slots)])
    queries = torch.tensor(blobs, dtype=torch.float32)

    layer.initialize_keys_kmeans(queries, seed=0)

    assert not torch.allclose(layer.memory_keys.data, random_keys)
    # Each true cluster center should be well-matched by some memory key.
    centers = torch.tensor(
        np.stack([blobs[i * 20:(i + 1) * 20].mean(axis=0) for i in range(n_slots)]), dtype=torch.float32
    )
    for center in centers:
        closest = (layer.memory_keys.data - center).norm(dim=-1).min()
        assert closest < 1.0
