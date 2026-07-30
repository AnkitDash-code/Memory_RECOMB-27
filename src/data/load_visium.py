import squidpy as sq


def load_visium_crop():
    """Small dataset for fast local iteration: 684 spots x 18078 genes."""
    return sq.datasets.visium_hne_adata_crop()


def load_visium_full():
    """Full benchmark dataset: 2688 spots x 18078 genes."""
    return sq.datasets.visium_hne_adata()
