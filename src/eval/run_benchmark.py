import json
import time
from pathlib import Path

import squidpy as sq
import torch

from src.data.load_visium import load_visium_crop, load_visium_full
from src.data.preprocess import preprocess
from src.eval.metrics import cluster_agreement, embedding_silhouette, spatial_coherence
from src.models.baseline_pca import run_baseline
from src.models.run_graphst import run_graphst
from src.models.train_memory_layer import cluster_from_embedding, train_memory_layer

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "benchmark_results.json"


def evaluate_method(adata, method_name, cluster_key, embedding_key, wall_time_s):
    labels = adata.obs[cluster_key].to_numpy()
    embedding = adata.obsm[embedding_key]
    return {
        "method": method_name,
        "n_clusters": int(len(set(labels))),
        "silhouette": embedding_silhouette(embedding, labels),
        "spatial_coherence_morans_i": spatial_coherence(adata, cluster_key)["mean"],
        "wall_time_s": round(wall_time_s, 3),
    }


def run_for_dataset(name, loader, device):
    print(f"\n=== {name} ===")
    raw_adata = loader()
    adata = preprocess(loader())

    start = time.time()
    adata = run_baseline(adata)
    baseline_time = time.time() - start
    baseline_result = evaluate_method(adata, "Scanpy PCA+Leiden (baseline)", "leiden", "X_pca", baseline_time)
    n_clusters = baseline_result["n_clusters"]

    start = time.time()
    _, adata, history = train_memory_layer(adata, epochs=300, device=device)
    adata = cluster_from_embedding(adata, "X_memory_trained", "memory_cluster_trained")
    memory_time = time.time() - start
    memory_result = evaluate_method(
        adata, "EmbeddedMemoryLayer (trained)", "memory_cluster_trained", "X_memory_trained", memory_time
    )
    memory_result["final_recon_loss"] = history[-1]["recon_loss"]
    memory_result["final_median_entropy"] = history[-1]["median_entropy"]

    start = time.time()
    graphst_adata = run_graphst(raw_adata, n_clusters=n_clusters, device=device)
    sq.gr.spatial_neighbors(graphst_adata, n_rings=1, coord_type="grid", n_neighs=6)
    graphst_time = time.time() - start
    graphst_result = evaluate_method(graphst_adata, "GraphST", "domain", "emb", graphst_time)

    agreement = cluster_agreement(adata.obs["leiden"], adata.obs["memory_cluster_trained"])
    agreement_graphst = cluster_agreement(adata.obs["leiden"].to_numpy(), graphst_adata.obs["domain"].to_numpy())

    for row in (baseline_result, memory_result, graphst_result):
        row["dataset"] = name
        print(row)
    print(f"ARI(baseline, memory_layer) = {agreement:.4f}  ARI(baseline, GraphST) = {agreement_graphst:.4f}")
    print("(agreement between methods, not accuracy -- none of these is ground truth)")

    return (
        [baseline_result, memory_result, graphst_result],
        {"baseline_vs_memory": agreement, "baseline_vs_graphst": agreement_graphst},
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []
    all_agreements = {}

    for name, loader in [("crop", load_visium_crop), ("full", load_visium_full)]:
        results, agreements = run_for_dataset(name, loader, device)
        all_results.extend(results)
        all_agreements[name] = agreements

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"results": all_results, "ari_agreements": all_agreements}, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
