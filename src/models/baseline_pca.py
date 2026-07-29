import json
import time
from pathlib import Path

import psutil
import scanpy as sc

from src.data.load_visium import load_visium_crop, load_visium_full
from src.data.preprocess import preprocess

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "baseline_timing.json"


def run_baseline(adata, n_pcs=50, resolution=1.0, n_clusters_target=None):
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata)
    if n_clusters_target is not None:
        from src.eval.metrics import search_leiden_resolution

        resolution = search_leiden_resolution(adata, None, n_clusters_target)
    sc.tl.leiden(adata, resolution=resolution)
    return adata


def time_baseline(name, adata):
    process = psutil.Process()
    ram_before_mb = process.memory_info().rss / 1024**2

    start = time.time()
    adata = run_baseline(adata)
    elapsed = time.time() - start

    ram_after_mb = process.memory_info().rss / 1024**2
    n_cells, n_genes = adata.shape
    n_clusters = adata.obs["leiden"].nunique()

    return {
        "dataset": name,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "wall_time_s": round(elapsed, 3),
        "peak_ram_mb": round(max(ram_before_mb, ram_after_mb), 1),
        "n_clusters_found": n_clusters,
    }


def main():
    datasets = {
        "visium_crop": preprocess(load_visium_crop()),
        "visium_full": preprocess(load_visium_full()),
    }

    results = []
    header = f"{'dataset':<14}{'n_cells':>10}{'n_genes':>10}{'wall_time_s':>14}{'peak_ram_mb':>14}{'n_clusters':>12}"
    print(header)
    for name, adata in datasets.items():
        row = time_baseline(name, adata)
        results.append(row)
        print(
            f"{row['dataset']:<14}{row['n_cells']:>10}{row['n_genes']:>10}"
            f"{row['wall_time_s']:>14}{row['peak_ram_mb']:>14}{row['n_clusters_found']:>12}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
