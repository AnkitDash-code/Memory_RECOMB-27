"""Loaders for the human DLPFC Visium benchmark (Maynard et al. 2021 / spatialLIBD).

Two sources, deliberately kept separate:

* `load_dlpfc_151673` reads the single slice recovered from the Kang et al. 2025
  benchmark's Zenodo release (raw Space Ranger output + metadata.tsv).
* `load_dlpfc_slice` / `load_all_dlpfc_slices` fetch any of the 12 slices as
  `.h5ad` from figshare (DOI 10.6084/m9.figshare.22004273, CC BY 4.0), which
  redistributes the spatialLIBD data used for CellCharter's benchmarking. This
  avoids needing R/Bioconductor to get the full 12-slice set.

Both attach the manual cortical-layer annotation to `obs['ground_truth_layer']`.
"""

import json
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import pandas as pd
import scanpy as sc

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DIR = DATA_ROOT / "dlpfc_151673"
SLICES_DIR = DATA_ROOT / "dlpfc_slices"

FIGSHARE_ARTICLE_ID = 22004273
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

ALL_DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

# Candidate column names for the manual layer annotation across sources.
_LAYER_COLUMNS = (
    "layer_guess_reordered",
    "layer_guess",
    "sce.layer_guess",  # column name used by the figshare/CellCharter .h5ad export
    "Layer",
    "ground_truth",
    "spatialLIBD",
)


def _resolve_layer_column(obs):
    for column in _LAYER_COLUMNS:
        if column in obs.columns:
            return column
    raise KeyError(
        f"No manual layer annotation found. Looked for {_LAYER_COLUMNS}; "
        f"available columns: {list(obs.columns)}"
    )


def load_dlpfc_151673(data_dir=DEFAULT_DIR):
    """DLPFC sample 151673 from the locally-extracted Kang et al. benchmark data.

    Raised as a clear error rather than auto-downloaded if absent.
    """
    data_dir = Path(data_dir)
    count_file = data_dir / "151673_filtered_feature_bc_matrix.h5"
    metadata_file = data_dir / "metadata.tsv"
    if not count_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"DLPFC 151673 data not found under {data_dir}. Expected "
            f"'{count_file.name}', 'spatial/', and 'metadata.tsv'."
        )

    adata = sc.read_visium(data_dir, count_file=count_file.name)
    adata.var_names_make_unique()

    metadata = pd.read_csv(metadata_file, sep="\t").set_index("barcode")
    adata.obs["ground_truth_layer"] = metadata.loc[adata.obs_names, "layer_guess_reordered"]
    return adata


def _figshare_download_urls():
    """Map sample id -> download URL, queried live rather than hardcoded."""
    with urlopen(FIGSHARE_API, timeout=60) as response:
        payload = json.load(response)
    return {Path(f["name"]).stem: f["download_url"] for f in payload["files"]}


def download_dlpfc_slices(samples=None, dest_dir=SLICES_DIR):
    """Download the requested DLPFC `.h5ad` slices from figshare if not cached.

    ~110MB per slice. Cached under data/ (gitignored) so this is a one-time cost.
    """
    samples = samples or ALL_DLPFC_SAMPLES
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    missing = [s for s in samples if not (dest_dir / f"{s}.h5ad").exists()]
    if not missing:
        return dest_dir

    urls = _figshare_download_urls()
    for sample in missing:
        if sample not in urls:
            raise KeyError(f"Sample {sample} not present in figshare record {FIGSHARE_ARTICLE_ID}")
        target = dest_dir / f"{sample}.h5ad"
        print(f"downloading DLPFC {sample} -> {target}")
        urlretrieve(urls[sample], target)
    return dest_dir


def load_dlpfc_slice(sample, dest_dir=SLICES_DIR, download=True):
    """Load one DLPFC slice by sample id (e.g. '151673') with ground truth attached."""
    dest_dir = Path(dest_dir)
    path = dest_dir / f"{sample}.h5ad"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found and download=False")
        download_dlpfc_slices([sample], dest_dir=dest_dir)

    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    adata.obs["ground_truth_layer"] = adata.obs[_resolve_layer_column(adata.obs)].astype(object)
    adata.obs["sample_id"] = sample
    return adata


def load_all_dlpfc_slices(samples=None, dest_dir=SLICES_DIR):
    """Yield (sample_id, adata) for each requested slice, one at a time.

    A generator on purpose: 12 slices held simultaneously would be several GB of
    RAM, and every consumer here processes them independently.
    """
    for sample in (samples or ALL_DLPFC_SAMPLES):
        yield sample, load_dlpfc_slice(sample, dest_dir=dest_dir)
