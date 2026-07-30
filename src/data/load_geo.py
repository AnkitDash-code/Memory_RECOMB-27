import squidpy as sq


def load_slideseqv2():
    """Real Slide-seqV2 mouse hippocampus dataset (squidpy.datasets.slideseqv2).

    Originally this module targeted GEO accession GSE129788, but that series is
    dissociated Drop-seq scRNA-seq (Ximerakis et al. 2019, aging mouse brain) with
    no spatial coordinates, so it cannot feed sq.gr.spatial_neighbors. slideseqv2()
    is a real, currently-shipping squidpy loader for actual Slide-seqV2 spatial
    data, requires no manual download, and has a different sparsity/platform
    profile than the Visium crop/full datasets.
    """
    return sq.datasets.slideseqv2()
