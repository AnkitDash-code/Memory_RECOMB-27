import json
from pathlib import Path

import torch

from src.data.load_visium import load_visium_crop, load_visium_full
from src.data.preprocess import get_pca_features, preprocess
from src.eval.vram_profile import profile_forward
from src.models.baseline_pca import time_baseline
from src.models.memory_layer import EmbeddedMemoryLayer

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "comparison_table.json"
VRAM_CEILING_MB = 5000


def profile_memory_layer(name, adata, device):
    # PCA-50 input, not raw gene expression: query_proj is feature_dim x
    # feature_dim, so raw ~15-18k gene dims would make it a ~200M-parameter
    # layer. PCA-50 also matches the baseline's own input representation.
    x = torch.tensor(get_pca_features(adata), dtype=torch.float32)
    layer = EmbeddedMemoryLayer(feature_dim=x.shape[1])

    elapsed, peak_mb = profile_forward(layer, x, device)
    n_cells, n_genes = adata.shape

    return {
        "method": "EmbeddedMemoryLayer",
        "dataset": name,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "wall_time_s": round(elapsed, 4),
        "peak_vram_mb": round(peak_mb, 1) if peak_mb is not None else None,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        "crop": preprocess(load_visium_crop()),
        "full": preprocess(load_visium_full()),
    }

    results = []
    for name, adata in datasets.items():
        baseline_row = time_baseline(f"baseline_{name}", adata.copy())
        results.append({
            "method": "Scanpy PCA+Leiden (baseline)",
            "dataset": name,
            "n_cells": baseline_row["n_cells"],
            "n_genes": baseline_row["n_genes"],
            "wall_time_s": baseline_row["wall_time_s"],
            "peak_ram_mb": baseline_row["peak_ram_mb"],
        })

        memory_row = profile_memory_layer(name, adata, device)
        if memory_row["peak_vram_mb"] is not None and memory_row["peak_vram_mb"] > VRAM_CEILING_MB:
            print(
                f"WARNING: {name} peak VRAM {memory_row['peak_vram_mb']:.1f} MB exceeds "
                f"{VRAM_CEILING_MB} MB ceiling -> move this dataset size to Colab"
            )
        results.append(memory_row)

    for row in results:
        print(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
