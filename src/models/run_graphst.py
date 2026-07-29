import GraphST
from GraphST.GraphST import GraphST as GraphSTModel


def run_graphst(adata, n_clusters, device, epochs=600):
    """Run the real GraphST package (Long et al., Nat Commun 2023) end to end.

    GraphST does its own HVG selection + normalize/log1p/scale and builds its
    own spatial graph from adata.obsm['spatial'] -- it expects raw counts, so
    callers should pass a freshly-loaded adata, not one already run through
    this repo's own preprocess() (which would double-normalize).

    method="leiden" in clustering() avoids GraphST's default 'mclust', which
    requires R/rpy2 -- not part of this project's stack.
    """
    adata = adata.copy()
    GraphST.preprocess(adata)
    GraphST.construct_interaction(adata)
    GraphST.add_contrastive_label(adata)
    GraphST.get_feature(adata)

    model = GraphSTModel(adata, device=device, epochs=epochs)
    adata = model.train()

    GraphST.clustering(adata, n_clusters=n_clusters, method="leiden")
    return adata
