"""Reproduces the LITERATURE_ARI_151673 numbers in run_dlpfc_benchmark.py.

These are NOT numbers copied from the Kang et al. 2025 (Nucleic Acids
Research) paper's text -- two independent attempts to read the paper's own
reported figures gave inconsistent numbers (0.498 vs. 0.515 for STAGATE), so
instead this computes ARI directly from that paper's own released per-spot
predictions for DLPFC sample 151673 (Zenodo record 15114362,
Benchmark_ST_analysis-master/2.SVG_indentified/Pred_label/151673_pred_label.csv)
against the real ground-truth layers in
Benchmark_ST_analysis-master/Dataset/DLPFC/151673/metadata.tsv
(spatialLIBD's layer_guess_reordered column).

Note: the released predictions CSV has an unresolved git merge conflict
baked into the archive (two `<<<<<<<`/`=======`/`>>>>>>>` sections); both
sides are byte-identical once their headers are aligned, so this uses
whichever side comes first without loss of information.

This script is not run as part of the normal pipeline -- it operates on the
external benchmark repo, not this project's own data -- and is kept here only
so the literature numbers quoted in outputs/logs/results_table.md are
reproducible, not just asserted.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def main(benchmark_repo_dir):
    benchmark_repo_dir = Path(benchmark_repo_dir)
    metadata = pd.read_csv(benchmark_repo_dir / "Dataset/DLPFC/151673/metadata.tsv", sep="\t")

    pred_path = benchmark_repo_dir / "2.SVG_indentified/Pred_label/151673_pred_label.csv"
    lines = pred_path.read_text().splitlines()
    head_marker = lines.index("<<<<<<< HEAD")
    separator = lines.index("=======")
    clean_lines = lines[head_marker + 2 : separator]  # skip conflict marker + duplicate header
    header = lines[head_marker + 1]
    from io import StringIO

    preds = pd.read_csv(StringIO("\n".join([header] + clean_lines)), index_col=0)

    assert len(metadata) == len(preds), "row count mismatch -- cannot assume aligned order"

    ground_truth = metadata["layer_guess_reordered"]
    mask = ground_truth.notna()

    results = {
        method: adjusted_rand_score(ground_truth[mask], preds[method][mask.values])
        for method in preds.columns
    }
    for method, ari in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{method:20s} ARI = {ari:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_literature_ari.py <path-to-extracted-Benchmark_ST_analysis-master>")
        sys.exit(1)
    main(sys.argv[1])
