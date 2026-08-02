"""Loader for the human breast cancer (10x Visium, Block A Section 1) benchmark.

This is the exact dataset GraphST's own paper (Long et al., Nat Commun 2023,
PMC9977836) evaluates: 3798 spots, 36601 genes, pathologist-annotated from
H&E + marker-gene expression into **20 regions** (ARI 0.54-0.57 reported).
Of the three Phase-C platforms this project considered, it is the only one
with a published, directly comparable GraphST ARI -- see
`outputs/logs/stage2_progress.md` for the literature check that established
this ordering.

**Provenance, traced rather than assumed.** The raw 10x Space Ranger output
is public (10xgenomics.com/datasets/human-breast-cancer-block-a-section-1),
but the 20-region pathologist annotation is not shipped there -- it comes
from Kang et al. 2025's benchmark (the same paper this project's DLPFC data
and literature-ARI numbers come from, `src/eval/compute_literature_ari.py`).
Kang et al.'s own code (`Benchmark_ST_analysis-master.zip`, Zenodo 15114362)
reads it from a `metadata.tsv` with a `fine_annot_type` column
(`utils_for_all.py::get_adata`, dataset="Breast_cancer") -- but that zip only
ships DLPFC's raw data, not breast cancer's. Cross-referencing that expected
schema against Kang et al.'s companion Figshare project
(figshare.com/projects/Benchmark_ST_analysis/234116, article 28200299,
"10X Visium") found one file group with the exact matching schema (`ID`,
`annot_type`, `fine_annot_type` columns; 3798 rows; 20 unique
`fine_annot_type` values) plus `aligned_fiducials.jpg` /
`detected_tissue_image.jpg` -- standard 10x Space Ranger QC images present on
no other file group in that project, confirming this is a distinct raw
sample rather than another DLPFC slice. Verified by direct download and
inspection, not inferred from file naming alone.
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import scanpy as sc

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DIR = DATA_ROOT / "breast_cancer"

# figshare article 28200299 ("10X Visium"), part of Kang et al. 2025's
# benchmark project (figshare.com/projects/Benchmark_ST_analysis/234116).
# File IDs verified by download: metadata.tsv has the annot_type/
# fine_annot_type columns Kang et al.'s own code expects for this dataset,
# and the h5/spatial files load as a normal Visium sample (3798 spots).
_FIGSHARE_FILES = {
    "filtered_feature_bc_matrix.h5": 51654728,
    "metadata.tsv": 51654650,
    "spatial/scalefactors_json.json": 51654656,
    "spatial/tissue_hires_image.png": 51654716,
    "spatial/tissue_lowres_image.png": 51654665,
    "spatial/tissue_positions_list.csv": 51654668,
}
_FIGSHARE_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{file_id}"

N_REGIONS = 20  # matches GraphST's own evaluation (fine_annot_type)


def download_breast_cancer(dest_dir=DEFAULT_DIR):
    """Fetch the raw counts, spatial files, and annotation if not cached."""
    dest_dir = Path(dest_dir)
    (dest_dir / "spatial").mkdir(parents=True, exist_ok=True)

    for relative_name, file_id in _FIGSHARE_FILES.items():
        target = dest_dir / relative_name
        if target.exists():
            continue
        print(f"downloading breast cancer {relative_name} -> {target}")
        urlretrieve(_FIGSHARE_DOWNLOAD_URL.format(file_id=file_id), target)
    return dest_dir


def load_breast_cancer(dest_dir=DEFAULT_DIR, download=True):
    """Load the breast cancer Visium sample with ground truth attached.

    Ground truth is the 20-category `fine_annot_type` (obs['ground_truth_region']),
    matching GraphST's own evaluation. The coarser 4-category `annot_type`
    (Healthy / Invasive / Surrounding tumor / Tumor) is also kept
    (obs['ground_truth_coarse']) since some methods report against it instead.
    """
    dest_dir = Path(dest_dir)
    if not (dest_dir / "filtered_feature_bc_matrix.h5").exists():
        if not download:
            raise FileNotFoundError(f"Breast cancer data not found under {dest_dir}")
        download_breast_cancer(dest_dir)

    adata = sc.read_visium(dest_dir, count_file="filtered_feature_bc_matrix.h5", load_images=True)
    adata.var_names_make_unique()

    metadata = pd.read_csv(dest_dir / "metadata.tsv", sep="\t", index_col=0)
    adata.obs["ground_truth_region"] = metadata.loc[adata.obs_names, "fine_annot_type"]
    adata.obs["ground_truth_coarse"] = metadata.loc[adata.obs_names, "annot_type"]
    return adata
