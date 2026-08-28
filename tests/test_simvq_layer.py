import torch

from src.models.simvq_layer import SimVQMemoryAutoencoder, SimVQMemoryLayer


def test_memory_keys_property_reflects_current_transform():
    torch.manual_seed(0)
    layer = SimVQMemoryLayer(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=2)
    keys_before = layer.memory_keys.clone()
    # identity init means keys start equal to the fixed basis
    assert torch.allclose(keys_before, layer.key_basis, atol=1e-5)

    with torch.no_grad():
        layer.key_transform.weight.add_(0.1)
    keys_after = layer.memory_keys
    assert not torch.allclose(keys_before, keys_after)  # property recomputes, doesn't cache stale keys


def test_key_basis_is_a_fixed_buffer_not_a_trainable_parameter():
    layer = SimVQMemoryLayer(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=2)
    param_names = {name for name, _ in layer.named_parameters()}
    assert "key_basis" not in param_names
    buffer_names = {name for name, _ in layer.named_buffers()}
    assert "key_basis" in buffer_names


def test_gradient_flows_to_shared_transform_not_to_fixed_basis():
    """The whole point of SimVQ: gradient updates the ONE shared linear
    transform (touching every key row simultaneously), not individual key
    rows independently. key_basis being a buffer already makes it
    impossible for gradient to reach it directly (no .grad attribute
    exists on a non-leaf buffer at all); this test confirms key_transform
    DOES receive a real, nonzero gradient, i.e. the reparameterization is
    actually wired into the loss.
    """
    torch.manual_seed(0)
    model = SimVQMemoryAutoencoder(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=1)
    x = torch.randn(10, 6)
    from src.models.memory_layer import normalized_adjacency
    import scipy.sparse as sp
    import numpy as np

    def _ring(n):
        rows = list(range(n)) + list(range(n))
        cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    adjacency = normalized_adjacency(_ring(10))
    reconstruction, embedding, addresses = model(x, adjacency)
    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()

    assert model.memory.key_transform.weight.grad is not None
    assert torch.any(model.memory.key_transform.weight.grad != 0)


def test_forward_shapes_and_valid_simplex():
    torch.manual_seed(0)
    import scipy.sparse as sp
    import numpy as np
    from src.models.memory_layer import normalized_adjacency

    def _ring(n):
        rows = list(range(n)) + list(range(n))
        cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    model = SimVQMemoryAutoencoder(feature_dim=7, memory_slots=6, memory_dim=5, hidden_dim=9, n_hops=2)
    x = torch.randn(14, 7)
    adjacency = normalized_adjacency(_ring(14))
    reconstruction, embedding, addresses = model(x, adjacency)
    assert reconstruction.shape == x.shape
    assert embedding.shape == (14, 5)
    assert addresses.shape == (14, 6)
    assert torch.allclose(addresses.sum(dim=-1), torch.ones(14), atol=1e-5)


def test_no_nans_through_forward_and_backward():
    torch.manual_seed(0)
    import scipy.sparse as sp
    import numpy as np
    from src.models.memory_layer import normalized_adjacency

    def _ring(n):
        rows = list(range(n)) + list(range(n))
        cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    model = SimVQMemoryAutoencoder(feature_dim=8, memory_slots=6, memory_dim=5, hidden_dim=10, n_hops=3)
    x = torch.randn(20, 8)
    adjacency = normalized_adjacency(_ring(20))
    reconstruction, embedding, addresses = model(x, adjacency)
    assert not torch.any(torch.isnan(reconstruction))
    assert not torch.any(torch.isnan(embedding))
    assert not torch.any(torch.isnan(addresses))
    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert not torch.any(torch.isnan(p.grad))
