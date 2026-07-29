import math

import torch

from src.models.memory_layer import (
    EmbeddedMemoryAutoencoder,
    EmbeddedMemoryLayer,
    attention_entropy,
    connectivities_to_edge_index,
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
