"""Stage 3 of the domain-scale-vs-propagation-depth follow-up plan: an
ADAPTIVE per-spot propagation-depth gate, balanced via loss-free bias
updates (DeepSeek-AI, "Auxiliary-Loss-Free Load Balancing Strategy for
Mixture-of-Experts", arXiv 2408.15664) instead of an auxiliary loss term.

Only reached because Stage 1 (a FIXED, non-adaptive monotone map) failed its
DLPFC threshold -- per the plan's own gating rule, this is the fallback for
"a more flexible gate is worth trying", not a default first choice.

Why this is a genuinely different mechanism from Phase D's adaptive_hops,
not a retry of it. Phase D tried exactly two variants of a per-spot learned
depth gate:
  (a) no regularizer at all -> collapsed to depth 0 (reconstruction loss has
      zero incentive to use propagation, since unsmoothed data always
      reconstructs more easily)
  (b) an AUXILIARY LOSS term (lambda_hop_usage, reusing usage_entropy on the
      gate weights) -> did prevent collapse, but underperformed the fixed
      baseline with the highest variance of any config tested (0.335 vs
      0.504 fixed n_hops, DLPFC).
Both variants route a balancing signal through backpropagation -- variant
(b)'s auxiliary loss creates exactly the "interference gradient" problem
DeepSeek's paper diagnoses: the balancing loss and the task loss compete for
the same gradient, degrading whichever the balancing term doesn't directly
serve. Loss-free balancing removes gradient from the balancing mechanism
entirely: routing bias is updated by a separate, non-differentiable rule
based on measured usage, never by backprop. If this still collapses or
still underperforms, that's evidence the routing SIGNAL itself (whatever a
per-spot query can encode about local domain scale) isn't informative here
-- not that the balancing mechanism was the problem, since this test
removes that specific confound.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.memory_layer import address_distribution


class LossFreeGatedMemoryLayer(nn.Module):
    """SpatialAddressMemoryLayer's addressing, with a learned per-spot gate
    over propagation depths 0..n_hops, balanced via a bias term updated
    OUTSIDE the gradient graph (call update_bias() after each optimizer
    step -- never inside the forward/backward pass).
    """

    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
        attention_fn="softmax",
        bias_update_rate=0.01,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.n_hops = n_hops
        self.temperature = temperature
        self.attention_fn = attention_fn
        self.bias_update_rate = bias_update_rate

        n_buckets = n_hops + 1
        self.hop_gate = nn.Linear(memory_dim, n_buckets)
        # Routing bias: NOT an nn.Parameter. Updated by update_bias() via a
        # fixed-step rule based on measured usage, never by .backward().
        self.register_buffer("depth_bias", torch.zeros(n_buckets))
        self._last_usage = None  # populated each forward() for update_bias() to read

    def forward(self, x, adjacency=None):
        queries = self.encoder(x)
        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        A0 = address_distribution(attn_scores, self.attention_fn, dim=-1)

        gate_logits = self.hop_gate(queries) + self.depth_bias  # bias added to logits, not to the loss
        gate_weights = F.softmax(gate_logits, dim=-1)  # (N, n_hops+1)

        if adjacency is not None:
            depths = [A0]
            current = A0
            for _ in range(self.n_hops):
                current = torch.sparse.mm(adjacency, current)
                depths.append(current)
            depth_stack = torch.stack(depths, dim=1)  # (N, n_hops+1, memory_slots)
            propagated = (depth_stack * gate_weights.unsqueeze(-1)).sum(dim=1)
        else:
            propagated = A0

        # Recorded for the (non-differentiable) bias update, done by the
        # caller after optimizer.step() -- detached so this bookkeeping can
        # never itself be part of the computation graph.
        self._last_usage = gate_weights.detach().mean(dim=0)

        embedding = torch.matmul(propagated, self.memory_values)
        return embedding, propagated, gate_weights

    @torch.no_grad()
    def update_bias(self):
        """Loss-free balancing update (DeepSeek arXiv 2408.15664): nudge each
        bucket's bias down if it was overloaded relative to uniform, up if
        underloaded, by a small FIXED step -- never a function of the loss
        or any gradient. Call once per training step, after optimizer.step(),
        using the usage measured during that step's forward pass.
        """
        if self._last_usage is None:
            return
        n_buckets = self.depth_bias.shape[0]
        target = 1.0 / n_buckets
        overloaded = self._last_usage > target
        self.depth_bias[overloaded] -= self.bias_update_rate
        self.depth_bias[~overloaded] += self.bias_update_rate


class LossFreeGatedMemoryAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
        attention_fn="softmax",
        bias_update_rate=0.01,
    ):
        super().__init__()
        self.memory = LossFreeGatedMemoryLayer(
            feature_dim=feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
            attention_fn=attention_fn,
            bias_update_rate=bias_update_rate,
        )
        self.decoder = nn.Linear(memory_dim, feature_dim)

    def forward(self, x, adjacency=None):
        embedding, addresses, gate_weights = self.memory(x, adjacency)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, addresses, gate_weights

    def update_bias(self):
        self.memory.update_bias()
