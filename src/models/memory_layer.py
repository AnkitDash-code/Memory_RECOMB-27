import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15, sparsemax


def address_distribution(scores, attention_fn="softmax", dim=-1):
    """Map raw address scores to a probability simplex, with a choice of
    normalization. All three are valid drop-in alternatives for the same
    address-distribution role in SpatialAddressMemoryLayer:

      * "softmax"  -- dense, every slot gets nonzero (if tiny) weight
      * "entmax15" -- 1.5-entmax (Tsallis alpha=1.5): sparse, differentiable,
                      a soft middle ground between softmax and sparsemax
      * "sparsemax"-- Euclidean projection onto the simplex: exact zeros for
                      low-scoring slots, can produce hard zero gradients for
                      pruned slots early in training (a known sparsemax
                      failure mode, not a bug if slot usage looks unstable)

    Motivation for the sparse variants: a dense softmax always gives every
    memory slot *some* weight, which can blur the address distribution
    (and thus the propagated embedding) across ambiguous spots even after
    training. A sparse projection forces each spot to commit to a small
    subset of slots, which may sharpen layer boundaries the same way
    expression-weighted adjacency (Stage 13) does for propagation weights,
    but acting on the address itself rather than the graph.
    """
    if attention_fn == "softmax":
        return F.softmax(scores, dim=dim)
    if attention_fn == "entmax15":
        return entmax15(scores, dim=dim)
    if attention_fn == "sparsemax":
        return sparsemax(scores, dim=dim)
    raise ValueError(f"Unknown attention_fn: {attention_fn!r}")


class EmbeddedMemoryLayer(nn.Module):
    def __init__(self, feature_dim, memory_slots=512, memory_dim=128):
        super().__init__()
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, feature_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.query_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        queries = self.query_proj(x)
        attn_scores = torch.matmul(queries, self.memory_keys.T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, self.memory_values), attn_weights


def attention_entropy(attn_weights):
    """Per-row entropy of the attention distribution, in nats.

    High per-row entropy means an individual spot is smeared across many slots
    (a mushy assignment). This is a diagnostic, not something to maximize --
    see usage_entropy for the quantity that actually prevents collapse.
    """
    eps = 1e-12
    return -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)


def usage_entropy(attn_weights):
    """Entropy of the MARGINAL slot-usage distribution (mean over spots), in nats.

    This is the quantity that prevents slot collapse, and it is a different
    thing from per-row entropy -- a distinction that matters in practice:

      * per-row entropy high  -> each spot spread thinly over many slots
                                 (soft, uninformative assignments)
      * usage entropy high    -> across the dataset, all slots get used,
                                 while any individual spot may still commit
                                 confidently to one slot

    We want the second, not the first. Maximizing usage entropy is the same
    load-balancing / equipartition idea used to stop codebook collapse in
    VQ-VAE-style models and expert collapse in mixture-of-experts routing.

    Without it, an MSE reconstruction objective has a strong early optimum:
    route every spot to a single slot that decodes to the dataset mean. That
    is exactly the failure observed here (slots_used=1, ARI=0.0) before this
    term was added.
    """
    eps = 1e-12
    usage = attn_weights.mean(dim=0)
    return -(usage * torch.log(usage + eps)).sum()


def key_cosine_similarity(memory_keys):
    """Mean pairwise cosine similarity between memory_keys rows (off-diagonal).

    A diagnostic distinct from usage_entropy: usage_entropy can look healthy
    (every slot gets used) while the KEY VECTORS themselves have collapsed to
    be nearly identical -- a subtler, "quiet" form of codebook degeneracy that
    usage statistics alone would not catch, since a spot can still be routed
    to different (but near-duplicate) slots. Low mean similarity indicates a
    well-spread codebook; a value approaching 1 indicates collapse. Added as
    the instrumentation an external review recommended be in place *before*
    running any new loss term that touches the codebook (e.g. a contrastive
    term), rather than diagnosing a failure after the fact -- exactly what
    happened with Stage 3's NB/ZINB + contrastive result, which was never
    decomposed into which change actually caused the regression.
    """
    normed = F.normalize(memory_keys, dim=-1)
    sim = normed @ normed.T
    n = sim.shape[0]
    off_diagonal_sum = sim.sum() - torch.diagonal(sim).sum()
    return off_diagonal_sum / (n * (n - 1))


def contrastive_address_loss(attn_weights, attn_corrupted):
    """Penalize agreement between real and feature-corrupted address assignments.

    GraphST and MAEST both report that a contrastive/denoising term is needed to
    stop the representation collapsing. Here the discrimination is done in
    ADDRESS space (not raw embeddings): a spot's address under real features
    should NOT match its address when the features are shuffled across spots.
    Implemented as the mean dot-product similarity between the two
    distributions, which is minimized.

    Originally introduced in Stage 3 (train_count_model.py) bundled together
    with an NB/ZINB reconstruction likelihood, and rejected as part of that
    combination without isolating which change caused the regression. Moved
    here so it can be tested on its own, on top of the winning MSE model, as
    an independent experiment -- see train_spatial_address_model's
    lambda_contrastive parameter.
    """
    return (attn_weights * attn_corrupted).sum(dim=-1).mean()


class EmbeddedMemoryAutoencoder(nn.Module):
    """Trainable wrapper around EmbeddedMemoryLayer.

    EmbeddedMemoryLayer alone has no loss signal: nothing pushes memory_keys
    or memory_values away from their random init. This adds a linear decoder
    back to feature space so the layer can be trained with a reconstruction
    objective, exposing the underlying memory-addressing mechanism to actual
    gradient signal instead of random output.
    """

    def __init__(self, feature_dim, memory_slots=512, memory_dim=128):
        super().__init__()
        self.memory = EmbeddedMemoryLayer(feature_dim, memory_slots, memory_dim)
        self.decoder = nn.Linear(memory_dim, feature_dim)

    def forward(self, x):
        embedding, attn_weights = self.memory(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, attn_weights


def spatial_smoothness_loss(embedding, edge_index, edge_weight=None):
    """Mean squared distance between embeddings of spatially-connected spots.

    edge_index: (2, n_edges) long tensor of (row, col) indices from the
    spatial connectivity graph. This is the paper's "memory-addressing
    replacing message passing" mechanism in the loss: instead of an explicit
    GNN layer propagating features along the spatial graph, neighboring spots
    are only encouraged (via this penalty) to land on similar memory
    addresses, with all cross-spot mixing happening through the shared
    memory bank rather than direct neighbor aggregation.
    """
    row, col = edge_index
    diff = embedding[row] - embedding[col]
    sq_dist = (diff**2).sum(dim=-1)
    if edge_weight is not None:
        sq_dist = sq_dist * edge_weight
    return sq_dist.mean()


def connectivities_to_edge_index(connectivities):
    """Convert an adata.obsp['spatial_connectivities'] sparse matrix to
    (edge_index, edge_weight) torch tensors."""
    coo = connectivities.tocoo()
    edge_index = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
    edge_weight = torch.tensor(coo.data, dtype=torch.float32)
    return edge_index, edge_weight


def normalized_adjacency(connectivities, device=None):
    """Row-normalized adjacency with self-loops, D^-1 (A + I), as a sparse tensor.

    Row-normalized (rather than symmetric D^-1/2 A D^-1/2) on purpose: rows must
    sum to 1 so that propagating a probability distribution keeps it a valid
    probability distribution -- see SpatialAddressMemoryLayer.
    """
    import scipy.sparse as sp

    adjacency = sp.csr_matrix(connectivities)
    adjacency = adjacency + sp.eye(adjacency.shape[0], format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    normalized = sp.diags(1.0 / degree) @ adjacency

    coo = normalized.tocoo()
    indices = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    sparse = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
    return sparse.to(device) if device is not None else sparse


def expression_weighted_adjacency(connectivities, features, device=None):
    """Same row-normalized D^-1(A+I) adjacency as normalized_adjacency, but each
    structural edge (i, j) is additionally reweighted by expression similarity:
    exp(-||x_i - x_j||^2 / (2 * sigma^2)), sigma set by the median heuristic
    (median squared edge distance) so no dataset-specific constant is needed.

    Motivation: pure spatial propagation blurs the address distribution across
    a boundary even when a neighbor is transcriptionally a different domain --
    exactly the failure mode on DLPFC's ambiguous subject-3 layer borders.
    Weighting edges down when the neighbor's expression is dissimilar lets
    propagation respect transcriptional boundaries, not just spatial adjacency.
    Self-loops are added at full weight 1 (a spot is maximally similar to
    itself) before row-normalizing, same as normalized_adjacency.

    features must be a numpy array (n_spots, n_features), e.g. from
    get_hvg_features(adata) -- computed once, not a learned/differentiable
    weighting.
    """
    import scipy.sparse as sp

    adjacency = sp.csr_matrix(connectivities)
    n = adjacency.shape[0]
    coo = adjacency.tocoo()
    row, col, data = coo.row, coo.col, coo.data

    diffs = features[row] - features[col]
    dists_sq = np.sum(diffs**2, axis=1)
    positive = dists_sq[dists_sq > 0]
    sigma_sq = np.median(positive) if positive.size > 0 else 1.0
    if sigma_sq <= 0:
        sigma_sq = 1.0
    expr_weight = np.exp(-dists_sq / (2 * sigma_sq))
    weighted_data = data * expr_weight

    weighted = sp.coo_matrix((weighted_data, (row, col)), shape=adjacency.shape).tocsr()
    weighted = weighted + sp.eye(n, format="csr")
    degree = np.asarray(weighted.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    normalized = sp.diags(1.0 / degree) @ weighted

    result = normalized.tocoo()
    indices = torch.tensor(np.stack([result.row, result.col]), dtype=torch.long)
    values = torch.tensor(result.data, dtype=torch.float32)
    sparse = torch.sparse_coo_tensor(indices, values, result.shape).coalesce()
    return sparse.to(device) if device is not None else sparse


def expression_similarity_edge_weights(connectivities, features, device=None):
    """Return structural edges weighted by expression similarity.

    This is kept separate from :func:`expression_weighted_adjacency` because
    the new address-coherence objective is an ablation of the loss, not a
    replacement for the graph used by the existing model.  The returned edge
    weights are symmetric when the input graph is symmetric and are not
    row-normalized.
    """
    import scipy.sparse as sp

    features = np.asarray(features, dtype=np.float32)
    adjacency = sp.coo_matrix(connectivities)
    mask = (adjacency.row != adjacency.col) & (adjacency.data > 0)
    row, col, data = adjacency.row[mask], adjacency.col[mask], adjacency.data[mask]
    if row.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty((0,), dtype=torch.float32, device=device)
        return edge_index, edge_weight

    diffs = features[row] - features[col]
    distances_sq = np.sum(diffs**2, axis=1)
    positive = distances_sq[distances_sq > 0]
    sigma_sq = np.median(positive) if positive.size else 1.0
    sigma_sq = max(float(sigma_sq), 1e-12)
    weights = data.astype(np.float32, copy=False) * np.exp(
        -distances_sq / (2.0 * sigma_sq)
    ).astype(np.float32)
    edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def address_spatial_coherence_loss(addresses, edge_index, edge_weight=None):
    """Penalize address discontinuity on expression-similar spatial edges.

    Unlike the concat-fusion mechanism, this term only changes the objective.
    Keeping it as a standalone function makes the loss-only and
    concat-plus-loss ablations explicit and testable.
    """
    if edge_index.numel() == 0:
        return addresses.sum() * 0.0
    row, col = edge_index
    squared_distance = ((addresses[row] - addresses[col]) ** 2).sum(dim=-1)
    if edge_weight is None:
        return squared_distance.mean()
    edge_weight = edge_weight.to(device=addresses.device, dtype=addresses.dtype)
    denominator = edge_weight.sum().clamp_min(torch.finfo(addresses.dtype).eps)
    return (squared_distance * edge_weight).sum() / denominator


class SpatialAddressMemoryLayer(nn.Module):
    """Memory addressing where the ADDRESS -- not the feature -- is spatially propagated.

    The project's premise is that memory-addressing can replace message passing.
    The original EmbeddedMemoryLayer took that literally: spatial structure
    entered only as a soft penalty in the loss, so a spot's embedding never saw
    its neighbours at all, and it badly underperformed GNN methods that
    aggregate neighbour features directly.

    This layer keeps the premise but makes it work. Spot features are still
    never mixed across spots. What gets propagated over the spatial graph is the
    softmax address distribution -- "which memory slots am I?" -- so neighbouring
    spots are pushed toward shared slot identity while their expression profiles
    stay independent:

        q = encoder(x)                    per-spot only, no neighbour info
        A = softmax(q @ keys.T)           address distribution (rows sum to 1)
        A = (D^-1 (Adj + I)) A   x k      propagate ADDRESSES, k hops
        z = A @ values                    embedding

    Because the propagation matrix is row-stochastic and A's rows are a
    probability simplex, each propagation step is a convex combination of
    neighbouring distributions and A stays a valid simplex -- no renormalization
    needed. Multi-hop (k > 1) widens the receptive field, following MAEST's
    finding that combining one-hop and multi-hop views helps.

    Biologically this encodes the laminar prior directly: cortical layers are
    spatially contiguous bands, so adjacent spots should share domain identity
    even where their individual expression is noisy or dropout-heavy.

    adaptive_hops=True (Phase C follow-up, see cross_validate_adaptive_hops.py)
    replaces the single fixed n_hops with a PER-SPOT learned combination of all
    depths 0..n_hops. Motivated by a mechanistic finding on Phase C's breast
    cancer result: n_hops=4 was cross-validated on DLPFC, where every layer
    (min 166 spots) is comfortably larger than a 4-hop neighbourhood's ~60-spot
    reach -- but breast cancer's 20 pathologist-annotated regions average only
    190 spots, several as small as 28-53, i.e. SMALLER than a single 4-hop
    neighbourhood. A fixed global hop count that works for large laminar bands
    over-smooths small domains by construction, mixing in neighbouring, unrelated
    tissue. Letting each spot's own query (already encoding its expression
    profile) gate how many hops of propagation to trust lets large homogeneous
    regions keep using deep propagation while small/heterogeneous regions can
    fall back toward their own (0-hop) address -- without retuning n_hops per
    dataset, which would just be leakage spread across datasets instead of slices.
    """

    def __init__(
        self,
        feature_dim,
        memory_slots=512,
        memory_dim=128,
        hidden_dim=256,
        n_hops=2,
        temperature=1.0,
        feature_hops=0,
        latent_hops=0,
        attention_fn="softmax",
        adaptive_hops=False,
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
        self.adaptive_hops = adaptive_hops
        if adaptive_hops:
            # Per-spot softmax over which of the (n_hops + 1) propagation
            # depths [0, 1, ..., n_hops] to use, conditioned on that spot's
            # own query (pre-propagation, so it reflects its own expression,
            # not already-smoothed information).
            self.hop_gate = nn.Linear(memory_dim, n_hops + 1)
        # Two distinct hybrid variants, deliberately separated because WHERE the
        # neighbour aggregation happens turns out to matter enormously:
        #
        #   feature_hops : smooth RAW features before encoding
        #   latent_hops  : smooth the ENCODED representation, which is what
        #                  GraphST actually does (z = adj @ (feat @ W1))
        #
        # Conflating the two would make an ablation of "hybrid vs. pure"
        # misleading, since only the latter matches the reference method.
        # Both default to 0, i.e. the pure "addressing replaces message
        # passing" formulation.
        self.feature_hops = feature_hops
        self.latent_hops = latent_hops

    def initialize_keys_kmeans(self, queries, seed=0):
        """Replace the random memory_keys init with k-means centroids of the
        encoder's own (still-random-weight) queries on the real data.

        Standard practice in VQ-style codebook methods (initializing the
        codebook from the data manifold rather than small random noise) to
        avoid a poor starting geometry. Motivated here by two documented
        problems: total slot collapse was observed before the usage-entropy
        fix (Stage 2), and per-seed ARI variance has remained high even after
        that fix and after cross-validating memory_slots (Stage 8) -- both
        consistent with the optimization being sensitive to where the
        codebook starts, which this targets directly.
        """
        from sklearn.cluster import KMeans

        with torch.no_grad():
            queries_np = queries.detach().cpu().numpy()
        n_clusters = self.memory_keys.shape[0]
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        kmeans.fit(queries_np)
        centers = torch.tensor(
            kmeans.cluster_centers_, dtype=self.memory_keys.dtype, device=self.memory_keys.device
        )
        self.memory_keys.data.copy_(centers)

    def forward(self, x, adjacency=None):
        if adjacency is not None and self.feature_hops:
            for _ in range(self.feature_hops):
                x = torch.sparse.mm(adjacency, x)

        queries = self.encoder(x)

        if adjacency is not None and self.latent_hops:
            for _ in range(self.latent_hops):
                queries = torch.sparse.mm(adjacency, queries)
        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        attn_weights = address_distribution(attn_scores, self.attention_fn, dim=-1)

        self.last_hop_gate_weights = None
        if adjacency is not None and self.adaptive_hops:
            depths = [attn_weights]
            current = attn_weights
            for _ in range(self.n_hops):
                current = torch.sparse.mm(adjacency, current)
                depths.append(current)
            depth_stack = torch.stack(depths, dim=1)  # (N, n_hops+1, memory_slots)
            gate_weights = F.softmax(self.hop_gate(queries), dim=-1)  # (N, n_hops+1)
            self.last_hop_gate_weights = gate_weights
            # Convex combination of simplices (gate sums to 1, each depth is
            # itself a valid simplex) is itself a valid simplex -- no
            # renormalization needed, same invariant the fixed-hop path keeps.
            propagated = (depth_stack * gate_weights.unsqueeze(-1)).sum(dim=1)
        else:
            propagated = attn_weights
            if adjacency is not None:
                for _ in range(self.n_hops):
                    propagated = torch.sparse.mm(adjacency, propagated)

        embedding = torch.matmul(propagated, self.memory_values)
        return embedding, propagated


class HopFusionMemoryLayer(nn.Module):
    """Concatenate address views from several physical-scale hop depths.

    The heterogeneity score is an ordinary input feature to ``fusion_mlp``.
    It never selects, gates, or softmaxes over hop depths.  This deliberately
    differs from ``SpatialAddressMemoryLayer(adaptive_hops=True)``, which is
    retained as a tested-and-rejected ablation.
    """

    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        max_hops=None,
        fusion_hidden_dim=128,
        fusion_depth=2,
        temperature=1.0,
        attention_fn="softmax",
    ):
        super().__init__()
        if max_hops is None:
            raise ValueError("max_hops must be supplied from the physical-scale config")
        if max_hops < 0:
            raise ValueError(f"max_hops must be >= 0, got {max_hops}")
        if fusion_hidden_dim < 1 or fusion_depth < 1:
            raise ValueError("fusion_hidden_dim and fusion_depth must be >= 1")

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, memory_dim) * 0.02)
        self.max_hops = int(max_hops)
        self.temperature = temperature
        self.attention_fn = attention_fn
        fusion_input_dim = (self.max_hops + 1) * memory_slots + 1
        fusion_layers = []
        input_dim = fusion_input_dim
        for _ in range(fusion_depth):
            fusion_layers.extend([nn.Linear(input_dim, fusion_hidden_dim), nn.ReLU()])
            input_dim = fusion_hidden_dim
        fusion_layers.append(nn.Linear(input_dim, memory_dim))
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        self.last_address_by_hop = None
        self.last_fusion_input = None

    @staticmethod
    def propagate(addresses, adjacency, hops):
        """Propagate an address simplex exactly ``hops`` times."""
        propagated = addresses
        if adjacency is None:
            return propagated
        for _ in range(int(hops)):
            propagated = torch.sparse.mm(adjacency, propagated)
        return propagated

    def forward(self, x, adjacency, heterogeneity_score):
        if heterogeneity_score is None:
            raise ValueError("heterogeneity_score is required for HopFusionMemoryLayer")
        queries = self.encoder(x)
        attn_scores = torch.matmul(queries, self.memory_keys.T) / self.temperature
        base_addresses = address_distribution(attn_scores, self.attention_fn, dim=-1)
        addr_by_hop = [
            self.propagate(base_addresses, adjacency, hops=h)
            for h in range(self.max_hops + 1)
        ]
        heterogeneity_score = heterogeneity_score.reshape(-1, 1).to(
            device=x.device, dtype=x.dtype
        )
        if heterogeneity_score.shape[0] != x.shape[0]:
            raise ValueError("heterogeneity_score must have one value per observation")
        fusion_input = torch.cat(addr_by_hop + [heterogeneity_score], dim=-1)
        self.last_address_by_hop = addr_by_hop
        self.last_fusion_input = fusion_input
        embedding = self.fusion_mlp(fusion_input)
        # The deepest address view is the natural slot-usage diagnostic and
        # preserves the two-tensor interface of the existing memory layer.
        return embedding, addr_by_hop[-1]


class SpatialAddressMemoryAutoencoder(nn.Module):
    """SpatialAddressMemoryLayer + a decoder back to gene-expression space.

    The decoder reconstructs the real (HVG) expression matrix rather than PCA
    scores, so the training signal is gene-level biological variation instead of
    an already-lossy linear compression.
    """

    def __init__(
        self,
        feature_dim,
        memory_slots=512,
        memory_dim=128,
        hidden_dim=256,
        n_hops=2,
        temperature=1.0,
        feature_hops=0,
        latent_hops=0,
        attention_fn="softmax",
        adaptive_hops=False,
    ):
        super().__init__()
        self.memory = SpatialAddressMemoryLayer(
            feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
            feature_hops=feature_hops,
            latent_hops=latent_hops,
            attention_fn=attention_fn,
            adaptive_hops=adaptive_hops,
        )
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x, adjacency=None):
        embedding, attn_weights = self.memory(x, adjacency)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, attn_weights


class HopFusionMemoryAutoencoder(nn.Module):
    """HopFusionMemoryLayer with the project's expression decoder."""

    def __init__(
        self,
        feature_dim,
        memory_slots=16,
        memory_dim=128,
        hidden_dim=256,
        max_hops=None,
        fusion_hidden_dim=128,
        fusion_depth=2,
        temperature=1.0,
        attention_fn="softmax",
    ):
        super().__init__()
        self.memory = HopFusionMemoryLayer(
            feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            max_hops=max_hops,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_depth=fusion_depth,
            temperature=temperature,
            attention_fn=attention_fn,
        )
        self.decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x, adjacency, heterogeneity_score):
        embedding, attn_weights = self.memory(x, adjacency, heterogeneity_score)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding, attn_weights


class SpatialAddressCountAutoencoder(nn.Module):
    """Address-propagation encoder with an NB/ZINB count decoder.

    Same addressing mechanism as SpatialAddressMemoryAutoencoder, but the
    decoder emits negative-binomial parameters over raw counts instead of a
    point estimate scored with MSE. Given 68-97% measured zeros, the Gaussian
    assumption behind MSE is a poor fit; NB/ZINB models overdispersion and
    (optionally) dropout explicitly.
    """

    def __init__(
        self,
        feature_dim,
        n_genes,
        memory_slots=64,
        memory_dim=128,
        hidden_dim=256,
        n_hops=4,
        temperature=1.0,
        zero_inflated=True,
    ):
        super().__init__()
        from src.models.count_losses import CountDecoder

        self.memory = SpatialAddressMemoryLayer(
            feature_dim,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            temperature=temperature,
        )
        self.decoder = CountDecoder(
            memory_dim, n_genes, hidden_dim=hidden_dim, zero_inflated=zero_inflated
        )

    def forward(self, x, library_size, adjacency=None):
        embedding, attn_weights = self.memory(x, adjacency)
        mu, theta, pi_logits = self.decoder(embedding, library_size)
        return (mu, theta, pi_logits), embedding, attn_weights


def main():
    import math

    from src.data.load_visium import load_visium_crop
    from src.data.preprocess import get_pca_features, preprocess

    adata = preprocess(load_visium_crop())
    x = torch.tensor(get_pca_features(adata), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = EmbeddedMemoryLayer(feature_dim=x.shape[1]).to(device)
    x = x.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _, attn_weights = layer(x)

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"peak VRAM: {peak_mb:.1f} MB")
    else:
        print("running on CPU; no VRAM to report")

    entropy = attention_entropy(attn_weights)
    max_entropy = math.log(layer.memory_keys.shape[0])
    print(f"attn_weights shape: {tuple(attn_weights.shape)}")
    print(f"median entropy: {entropy.median().item():.4f} (max possible: {max_entropy:.4f})")


if __name__ == "__main__":
    main()
