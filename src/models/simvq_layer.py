"""Stage 2 of the domain-scale-vs-propagation-depth follow-up plan: SimVQ-style
codebook reparameterization (Zhu, Li, Xin, Xia, Xu, ICCV 2025, arXiv 2411.02038,
"Addressing Representation Collapse in Vector Quantized Models with One Linear
Layer").

IMPORTANT CALIBRATION, checked before implementing this: SimVQ and the
rotation trick (Fifty et al., arXiv 2410.06424) both target a specific
failure mode of HARD vector quantization -- nearest-neighbour argmin
selection with a straight-through gradient estimator, where only the
selected codebook entry receives a gradient from a given sample, so entries
far from the data manifold never get updated ("disjoint codebook
optimization", dead codebook entries).

This project's SpatialAddressMemoryLayer does not do hard VQ. Addressing is
DENSE softmax attention over all memory_keys -- every key receives some
gradient from every sample, proportional to its attention weight. Checked
tonight's own baseline logs (outputs/logs/baseline_{dlpfc,breast_cancer}_
results.json, 60 fits total) before writing any code here: n_slots_used is
16/16 or 15/16 in every single fit, usage_entropy sits at 99.7-99.9% of its
theoretical maximum throughout, and key_cosine_similarity is already low
(-0.05 to +0.14, nowhere near the ~1.0 that would indicate collapse). The
existing lambda_usage regularizer already keeps this model's codebook near
full utilization. The specific problem SimVQ was designed to fix does not
appear to be present in this architecture.

Implementing and testing anyway, per the plan's own framing ("if utilization
improves but ARI doesn't, log it as a real but non-adopted result --
utilization was not the binding constraint") -- a correctly-calibrated null
test is still real evidence, not wasted effort. The adapted justification
for trying it here is weaker than the original paper's: SimVQ's one shared
linear transform over all key rows could still change how the keys move
relative to each other during optimization even without dead entries to
revive, but this is speculative, not the paper's own claim, and stated as
such.

Applied to memory_keys only (the addressing/lookup surface), not
memory_values -- SimVQ's claim is specifically about which codebook entries
get selected and how they're updated, i.e. the lookup surface, not the
readout surface, and keeping memory_values as a plain nn.Parameter isolates
exactly what's being tested.
"""

import torch
import torch.nn as nn

from src.models.memory_layer import address_distribution


class SimVQMemoryLayer(nn.Module):
    """SpatialAddressMemoryLayer's fixed-hop addressing, with memory_keys
    reparameterized as key_basis @ W (one shared learnable linear transform)
    instead of a directly-optimized nn.Parameter.
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
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        # Fixed random basis (not directly optimized -- SimVQ's point is that
        # ALL basis rows move together through the shared transform W, rather
        # than each key being independently, disjointly optimized).
        self.register_buffer("key_basis", torch.randn(memory_slots, memory_dim) * 0.02)
        self.key_transform = nn.Linear(memory_dim, memory_dim, bias=False)
        nn.init.eye_(self.key_transform.weight)  # start as identity: W=I means keys start = basis, same init scale as the baseline model

        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.n_hops = n_hops
        self.temperature = temperature
        self.attention_fn = attention_fn

    @property
    def memory_keys(self):
        """The effective codebook: basis rows pushed through the one shared
        linear transform. A property (not a stored Parameter) so every
        forward pass reflects the current state of key_transform.
        """
        return self.key_transform(self.key_basis)

    def forward(self, x, adjacency=None):
        queries = self.encoder(x)
        keys = self.memory_keys
        attn_scores = torch.matmul(queries, keys.T) / self.temperature
        A0 = address_distribution(attn_scores, self.attention_fn, dim=-1)

        propagated = A0
        if adjacency is not None:
            for _ in range(self.n_hops):
                propagated = torch.sparse.mm(adjacency, propagated)

        embedding = torch.matmul(propagated, self.memory_values)
        return embedding, propagated


class SimVQMemoryAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
        attention_fn="softmax",
    ):
        super().__init__()
        self.memory = SimVQMemoryLayer(
            feature_dim=feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
            attention_fn=attention_fn,
        )
        self.decoder = nn.Linear(memory_dim, feature_dim)

    def forward(self, x, adjacency=None):
        embedding, addresses = self.memory(x, adjacency)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, addresses
