import GraphST
from GraphST.GraphST import GraphST as GraphSTModel
from GraphST.preprocess import construct_interaction_KNN


def run_graphst(adata, n_clusters, device, epochs=600, random_seed=41, cluster=True, datatype="10X"):
    """Run the real GraphST package (Long et al., Nat Commun 2023) end to end.

    GraphST does its own HVG selection + normalize/log1p/scale and builds its
    own spatial graph from adata.obsm['spatial'] -- it expects raw counts, so
    callers should pass a freshly-loaded adata, not one already run through
    this repo's own preprocess() (which would double-normalize).

    random_seed is exposed (GraphST's own default is 41) so GraphST can be run
    across seeds too. Comparing our multi-seed mean against a single GraphST run
    would not characterize the gap honestly -- both sides need a variance.

    cluster=True keeps GraphST's own clustering (writing obs['domain']), which
    existing callers depend on. Pass cluster=False to skip it -- its Leiden path
    runs a slow resolution search -- when the caller is going to score the
    returned embedding with src/eval/clustering.py, the protocol applied
    uniformly to every method here.

    datatype="10X" (GraphST's own default) builds a DENSE pairwise spatial
    adjacency (`construct_interaction`) -- an O(n^2) memory allocation, fine
    at DLPFC/breast-cancer scale (~3-5k spots) but a confirmed
    `ArrayMemoryError` at Slide-seqV2 scale (~42k spots, 13GB dense matrix).
    Pass datatype="Slide" or "Stereo" (GraphST's own two large-N cases) to use
    `construct_interaction_KNN` instead, matching what GraphSTModel's own
    __init__ does internally for these datatypes -- must be selected here too
    since this wrapper pre-builds `adata.obsm['adj']` before constructing the
    model, and the model skips reconstruction when 'adj' already exists.
    """
    adata = adata.copy()
    GraphST.preprocess(adata)
    if datatype in ("Stereo", "Slide"):
        construct_interaction_KNN(adata)
    else:
        GraphST.construct_interaction(adata)
    GraphST.add_contrastive_label(adata)
    GraphST.get_feature(adata)

    model = GraphSTModel(adata, device=device, epochs=epochs, random_seed=random_seed, datatype=datatype)
    adata = model.train()

    if cluster:
        GraphST.clustering(adata, n_clusters=n_clusters, method="leiden")
    return adata
