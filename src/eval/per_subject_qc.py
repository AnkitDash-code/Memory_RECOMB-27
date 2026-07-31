"""Per-subject raw-data QC for the 12 DLPFC slices.

Motivation: three architecture-level hypotheses have now been tested and
specifically falsified on subject 3 (151673-676), the persistent weak point --
expression/morphology disagreement (Stage 16/17), address-space contrastive
regularization (Stage 15), and "the model lacks a laminar ordering" (Stage 18,
which found subject 3 already has the STRONGEST and most stable ordinal axis
of the three subjects, 0.824 +/- 0.02). Three specific falsifications is a
pattern, not bad luck, and points at a data-quality ceiling rather than an
architectural deficit.

This checks that directly, and it should have been checked long ago:
PROGRESS.md has been asserting that "no data-level explanation (sparsity,
layer proportions, spot count) has been found so far" -- but `data_stats.py`
only ever covered the mouse Visium and Slide-seq datasets, never DLPFC and
never per-subject. The claim was not backed by this measurement. Fixed here.

Metrics, all on RAW counts (pre-HVG, pre-normalization -- the point is data
quality as delivered, not after we have already conditioned it):
  * median library size (total counts per spot) -- sequencing depth
  * median genes detected per spot -- complementary depth measure
  * dropout rate (fraction of exact zeros in the count matrix)
  * spot count and spatial sampling density
  * fraction of spots lacking a manual layer annotation -- annotation quality,
    a distinct axis from sequencing quality and one that directly caps
    achievable ARI

Pure data statistics: no model, no training, no GPU.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.data.load_dlpfc import load_dlpfc_slice
from src.eval.analyze_multislice_variance import SUBJECTS

RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "per_subject_qc.json"


def _counts_matrix(adata):
    x = adata.X
    return x if sp.issparse(x) else sp.csr_matrix(x)


def run_slice(sample):
    adata = load_dlpfc_slice(sample)
    x = _counts_matrix(adata)

    library_size = np.asarray(x.sum(axis=1)).ravel()
    genes_detected = np.asarray((x > 0).sum(axis=1)).ravel()
    dropout_rate = 1.0 - (x.nnz / (x.shape[0] * x.shape[1]))

    coords = adata.obsm["spatial"]
    span_x = coords[:, 0].max() - coords[:, 0].min()
    span_y = coords[:, 1].max() - coords[:, 1].min()
    area = float(span_x) * float(span_y)
    density = adata.n_obs / area if area > 0 else float("nan")

    truth = adata.obs["ground_truth_layer"]
    unannotated_frac = float(truth.isna().mean())

    return {
        "sample": sample,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "median_library_size": float(np.median(library_size)),
        "median_genes_detected": float(np.median(genes_detected)),
        "dropout_rate": float(dropout_rate),
        "spot_density_per_1e6px": float(density * 1e6),
        "unannotated_fraction": unannotated_frac,
        "n_layers": int(truth.nunique()),
    }


def main():
    results = []
    header = (f"{'sample':<9}{'subj':<9}{'spots':>7}{'med_lib':>10}{'med_genes':>11}"
              f"{'dropout':>9}{'density':>9}{'unann%':>8}{'layers':>7}")
    print(header)
    print("-" * len(header))
    for subject, samples in SUBJECTS.items():
        for sample in samples:
            row = run_slice(sample)
            row["subject"] = subject
            results.append(row)
            print(f"{sample:<9}{subject:<9}{row['n_spots']:>7}"
                  f"{row['median_library_size']:>10.0f}{row['median_genes_detected']:>11.0f}"
                  f"{row['dropout_rate']:>9.4f}{row['spot_density_per_1e6px']:>9.2f}"
                  f"{row['unannotated_fraction'] * 100:>8.2f}{row['n_layers']:>7}", flush=True)

    print("\n=== per-subject means ===")
    keys = ["n_spots", "median_library_size", "median_genes_detected",
            "dropout_rate", "spot_density_per_1e6px", "unannotated_fraction"]
    per_subject = {}
    print(f"{'subject':<10}" + "".join(f"{k.split('_')[-1][:9]:>12}" for k in keys))
    for subject in SUBJECTS:
        rows = [r for r in results if r["subject"] == subject]
        summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        per_subject[subject] = summary
        print(f"{subject:<10}" + "".join(f"{summary[k]:>12.4f}" for k in keys))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"per_slice": results, "per_subject": per_subject}, indent=2)
    )
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
