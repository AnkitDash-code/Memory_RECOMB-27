import numpy as np
import scipy.sparse as sp
import torch

from src.models.loss_free_gated_layer import (
    LossFreeGatedMemoryAutoencoder,
    LossFreeGatedMemoryLayer,
)
from src.models.memory_layer import normalized_adjacency


def _ring(n):
    rows = list(range(n)) + list(range(n))
    cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_depth_bias_is_a_buffer_not_a_trainable_parameter():
    layer = LossFreeGatedMemoryLayer(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=3)
    param_names = {name for name, _ in layer.named_parameters()}
    assert "depth_bias" not in param_names
    buffer_names = {name for name, _ in layer.named_buffers()}
    assert "depth_bias" in buffer_names


def test_hop_gate_does_receive_gradient_unlike_stage_1s_fixed_certainty():
    """Deliberately the opposite invariant from Stage 1: this stage's gate IS
    supposed to be learned (that's the point -- adaptive, not fixed), only
    the BALANCING is loss-free. Confirm hop_gate.weight gets a real gradient.
    """
    torch.manual_seed(0)
    model = LossFreeGatedMemoryAutoencoder(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=2)
    x = torch.randn(10, 6)
    adjacency = normalized_adjacency(_ring(10))
    reconstruction, embedding, addresses, gate_weights = model(x, adjacency)
    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()
    assert model.memory.hop_gate.weight.grad is not None
    assert torch.any(model.memory.hop_gate.weight.grad != 0)


def test_update_bias_nudges_overloaded_buckets_down_and_underloaded_up():
    layer = LossFreeGatedMemoryLayer(feature_dim=4, memory_slots=3, memory_dim=3, hidden_dim=5, n_hops=3)
    n_buckets = 4
    # Fabricate a usage pattern: bucket 0 wildly overloaded, bucket 3 unused.
    layer._last_usage = torch.tensor([0.9, 0.05, 0.03, 0.02])
    bias_before = layer.depth_bias.clone()
    layer.update_bias()
    assert layer.depth_bias[0] < bias_before[0]  # overloaded -> bias decreases
    assert layer.depth_bias[3] > bias_before[3]  # underloaded -> bias increases


def test_update_bias_requires_no_grad_and_does_not_touch_gradients():
    torch.manual_seed(0)
    model = LossFreeGatedMemoryAutoencoder(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=2)
    x = torch.randn(10, 6)
    adjacency = normalized_adjacency(_ring(10))
    reconstruction, embedding, addresses, gate_weights = model(x, adjacency)
    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()
    grad_before = model.memory.hop_gate.weight.grad.clone()
    model.update_bias()  # must not raise, must not require an active grad context
    assert torch.equal(model.memory.hop_gate.weight.grad, grad_before)  # unaffected


def test_update_bias_repeated_calls_keep_bias_bounded_and_finite():
    layer = LossFreeGatedMemoryLayer(feature_dim=4, memory_slots=3, memory_dim=3, hidden_dim=5, n_hops=3)
    layer._last_usage = torch.tensor([0.9, 0.05, 0.03, 0.02])
    for _ in range(200):
        layer.update_bias()
    assert torch.all(torch.isfinite(layer.depth_bias))


def test_forward_shapes_and_valid_simplices():
    torch.manual_seed(0)
    layer = LossFreeGatedMemoryLayer(feature_dim=6, memory_slots=5, memory_dim=4, hidden_dim=8, n_hops=3)
    x = torch.randn(12, 6)
    adjacency = normalized_adjacency(_ring(12))
    embedding, addresses, gate_weights = layer(x, adjacency)
    assert embedding.shape == (12, 4)
    assert addresses.shape == (12, 5)
    assert gate_weights.shape == (12, 4)  # n_hops + 1
    assert torch.allclose(addresses.sum(dim=-1), torch.ones(12), atol=1e-5)
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(12), atol=1e-5)


def test_no_nans_through_forward_and_backward():
    torch.manual_seed(0)
    model = LossFreeGatedMemoryAutoencoder(feature_dim=8, memory_slots=6, memory_dim=5, hidden_dim=10, n_hops=3)
    x = torch.randn(20, 8)
    adjacency = normalized_adjacency(_ring(20))
    reconstruction, embedding, addresses, gate_weights = model(x, adjacency)
    assert not torch.any(torch.isnan(reconstruction))
    assert not torch.any(torch.isnan(gate_weights))
    loss = torch.nn.functional.mse_loss(reconstruction, x)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert not torch.any(torch.isnan(p.grad))
