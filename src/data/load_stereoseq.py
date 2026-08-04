"""Loader boundary for the Stereo-seq mouse olfactory-bulb holdout.

The repository does not currently vendor a Stereo-seq object.  Keeping the
loader path-based makes provenance explicit: the generalization runner can be
used with the exact h5ad exported for the chosen Stereo-seq OB study without
silently substituting a different tissue or resolution.
"""

from pathlib import Path

import scanpy as sc

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_STEREOSEQ_PATH = DATA_ROOT / "stereoseq_olfactory_bulb.h5ad"


def load_stereoseq_olfactory_bulb(path=DEFAULT_STEREOSEQ_PATH):
    """Load a local Stereo-seq olfactory-bulb AnnData object.

    The object must contain ``obsm['spatial']`` and expression data.  No
    supervised labels are required or inferred; the generalization protocol
    reports unsupervised metrics and marker-gene agreement only.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Stereo-seq OB data not found at {path}. Download/export the chosen "
            "Stereo-seq olfactory-bulb h5ad and pass it with --stereo-path."
        )
    if path.suffix.lower() != ".h5ad":
        raise ValueError(f"Expected an .h5ad Stereo-seq object, got {path}")
    adata = sc.read_h5ad(path)
    if "spatial" not in adata.obsm:
        raise KeyError("Stereo-seq AnnData must contain obsm['spatial'] coordinates")
    adata.uns.setdefault("platform", "stereoseq")
    return adata
