"""Apply one locked DLPFC Hop-Fusion config to new datasets.

This runner keeps the two claims separate:

* breast cancer is a same-platform domain-size test and includes a required
  contiguous-block holdout;
* Stereo-seq olfactory bulb is the physically distinct platform test and is
  scored with unsupervised metrics only.

No command in this module retunes the locked config.  Stereo-seq data must be
provided explicitly because the repository does not ship a particular OB
study's h5ad object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS, load_breast_cancer
from src.data.load_stereoseq import load_stereoseq_olfactory_bulb
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.eval.hop_fusion_protocol import (
    DEFAULT_LOCK_PATH,
    contiguous_spatial_block_masks,
    load_locked_hop_fusion_config,
)
from src.eval.metrics import embedding_silhouette, spatial_coherence
from src.models.run_graphst import run_graphst
from src.models.train_hop_fusion import train_hop_fusion_model
from src.models.train_spatial_address import train_spatial_address_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "hop_fusion_generalization.json"


def _score(adata, embedding, labels, marker_sets=None):
    adata = adata.copy()
    adata.obs["_hop_fusion_pred"] = np.asarray(labels).astype(str)
    result = {
        "silhouette": embedding_silhouette(embedding, labels),
        "spatial_coherence_morans_i": spatial_coherence(adata, "_hop_fusion_pred")["mean"],
    }
    if marker_sets:
        result["marker_gene_agreement"] = _marker_gene_agreement(adata, labels, marker_sets)
    return result


def _marker_gene_agreement(adata, labels, marker_sets):
    """Summarize whether supplied marker sets separate any predicted cluster.

    This is an unsupervised marker-separation diagnostic, not a supervised
    accuracy score.  Marker sets are supplied as ``{name: [gene, ...]}`` so
    the choice of biological markers remains visible in the output.
    """
    labels = np.asarray(labels).astype(str)
    categories = np.unique(labels)
    per_set = {}
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    for name, genes in marker_sets.items():
        positions = [adata.var_names.get_loc(gene) for gene in genes if gene in adata.var_names]
        if not positions:
            per_set[name] = None
            continue
        expression = matrix[:, positions].mean(axis=1)
        means = np.array([expression[labels == category].mean() for category in categories])
        overall = float(expression.mean())
        per_set[name] = {
            "genes_found": len(positions),
            "top_cluster": str(categories[int(np.argmax(means))]),
            "top_cluster_mean_minus_global": float(np.max(means) - overall),
        }
    values = [
        row["top_cluster_mean_minus_global"]
        for row in per_set.values()
        if row is not None
    ]
    return {
        "mean_top_cluster_mean_minus_global": float(np.mean(values)) if values else None,
        "per_marker_set": per_set,
    }


def _fusion_kwargs(config, platform):
    return {
        "platform": platform,
        "physical_radius_um": config["physical_radius_um"],
        "memory_slots": config["memory_slots"],
        "memory_dim": config["memory_dim"],
        "hidden_dim": config["hidden_dim"],
        "fusion_hidden_dim": config["fusion_hidden_dim"],
        "fusion_depth": config["fusion_depth"],
        "lambda_usage": config["lambda_usage"],
        "lambda_spatial_coherence": config["lambda_spatial_coherence"],
        "expression_weighted": config["expression_weighted"],
        "attention_fn": config["attention_fn"],
    }


def _evaluate_breast(config, seeds, device, epochs):
    raw = load_breast_cancer()
    adata = preprocess_hvg(raw.copy(), platform="visium")
    truth = adata.obs["ground_truth_region"].to_numpy()
    valid = adata.obs["ground_truth_region"].notna().to_numpy()
    coords = adata.obsm["spatial"]

    full_scores = []
    block_scores = []
    for seed in seeds:
        _, trained, _ = train_hop_fusion_model(
            adata.copy(), seed=seed, device=device, epochs=epochs,
            **_fusion_kwargs(config, "visium"),
        )
        labels = cluster_embedding(trained.obsm["X_hop_fusion"], N_REGIONS, coords=coords, refine=True)
        full_scores.append(float(adjusted_rand_score(truth[valid], labels[valid])))

    for block_name, masks in contiguous_spatial_block_masks(coords).items():
        per_seed = []
        for seed in seeds:
            _, trained, _ = train_hop_fusion_model(
                adata.copy(), seed=seed, train_mask=masks["train"],
                device=device, epochs=epochs, **_fusion_kwargs(config, "visium"),
            )
            labels = cluster_embedding(trained.obsm["X_hop_fusion"], N_REGIONS, coords=coords, refine=True)
            per_seed.append({
                "all_annotated_ari": float(adjusted_rand_score(truth[valid], labels[valid])),
                "validation_block_ari": float(
                    adjusted_rand_score(truth[masks["validation"]], labels[masks["validation"]])
                ),
                "validation_fraction": float(masks["validation"].mean()),
            })
        block_scores.append({"block": block_name, "per_seed": per_seed})

    fixed_scores = []
    for seed in seeds:
        _, trained, _ = train_spatial_address_model(
            adata.copy(),
            n_hops=config["fixed_baseline_n_hops"],
            memory_slots=config["memory_slots"],
            memory_dim=config["memory_dim"],
            hidden_dim=config["hidden_dim"],
            lambda_usage=config["lambda_usage"],
            expression_weighted=config["expression_weighted"],
            attention_fn=config["attention_fn"],
            seed=seed, device=device, epochs=epochs, verbose=False,
        )
        labels = cluster_embedding(trained.obsm["X_spatial_address"], N_REGIONS, coords=coords, refine=True)
        fixed_scores.append(float(adjusted_rand_score(truth[valid], labels[valid])))

    graphst_scores = []
    for seed in seeds:
        graphst_adata = run_graphst(
            raw.copy(), n_clusters=N_REGIONS, device=device,
            random_seed=seed, cluster=False,
        )
        labels = cluster_embedding(
            graphst_adata.obsm["emb"], N_REGIONS,
            coords=graphst_adata.obsm["spatial"], refine=True,
        )
        graph_truth = truth[adata.obs_names.get_indexer(graphst_adata.obs_names)]
        graphst_scores.append(float(adjusted_rand_score(graph_truth, labels)))

    return {
        "n_spots": int(adata.n_obs),
        "n_regions": N_REGIONS,
        "hop_fusion": {"per_seed_ari": full_scores, "mean": float(np.mean(full_scores))},
        "fixed_hop_comparator": {"per_seed_ari": fixed_scores, "mean": float(np.mean(fixed_scores))},
        "graphst": {"per_seed_ari": graphst_scores, "mean": float(np.mean(graphst_scores))},
        "within_sample_spatial_block_holdout": block_scores,
        "interpretation": "breast cancer tests the same-platform domain-size mismatch, not cross-platform generalization",
    }


def _evaluate_stereo(config, path, n_clusters, seeds, device, epochs, marker_sets):
    raw = load_stereoseq_olfactory_bulb(path)
    adata = preprocess_hvg(
        raw.copy(), coord_type="generic", platform="stereoseq"
    )
    coords = adata.obsm["spatial"]
    methods = {"hop_fusion": [], "fixed_hop_comparator": [], "graphst": []}
    for seed in seeds:
        _, trained, _ = train_hop_fusion_model(
            adata.copy(), seed=seed, device=device, epochs=epochs,
            **_fusion_kwargs(config, "stereoseq"),
        )
        labels = cluster_embedding(trained.obsm["X_hop_fusion"], n_clusters, coords=coords, refine=True)
        methods["hop_fusion"].append(_score(adata, trained.obsm["X_hop_fusion"], labels, marker_sets))

        _, fixed, _ = train_spatial_address_model(
            adata.copy(),
            n_hops=config["fixed_baseline_n_hops"],
            memory_slots=config["memory_slots"], memory_dim=config["memory_dim"],
            hidden_dim=config["hidden_dim"], lambda_usage=config["lambda_usage"],
            expression_weighted=config["expression_weighted"],
            attention_fn=config["attention_fn"], seed=seed, device=device,
            epochs=epochs, verbose=False,
        )
        fixed_labels = cluster_embedding(fixed.obsm["X_spatial_address"], n_clusters, coords=coords, refine=True)
        methods["fixed_hop_comparator"].append(
            _score(adata, fixed.obsm["X_spatial_address"], fixed_labels, marker_sets)
        )

        graphst_adata = run_graphst(
            raw.copy(), n_clusters=n_clusters, device=device,
            random_seed=seed, cluster=False, datatype="Stereo",
        )
        graph_labels = cluster_embedding(
            graphst_adata.obsm["emb"], n_clusters,
            coords=graphst_adata.obsm["spatial"], refine=True,
        )
        graph_score_adata = adata.copy()
        graph_score_adata.obsm["X_graphst"] = graphst_adata.obsm["emb"]
        methods["graphst"].append(
            _score(graph_score_adata, graphst_adata.obsm["emb"], graph_labels, marker_sets)
        )

    return {
        "n_spots": int(adata.n_obs),
        "n_clusters": n_clusters,
        "methods": methods,
        "interpretation": "Stereo-seq OB is evaluated with unsupervised metrics; GraphST has no supervised ARI for this platform",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo-path", required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--config", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-breast", action="store_true")
    parser.add_argument("--markers-json", help="JSON object mapping marker-set names to gene lists")
    args = parser.parse_args()

    config = load_locked_hop_fusion_config(args.config)
    marker_sets = json.loads(Path(args.markers_json).read_text()) if args.markers_json else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "config": str(Path(args.config).resolve()),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "breast_cancer": None if args.skip_breast else _evaluate_breast(
            config, args.seeds, device, args.epochs
        ),
        "stereoseq_olfactory_bulb": _evaluate_stereo(
            config, args.stereo_path, args.n_clusters, args.seeds,
            device, args.epochs, marker_sets,
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
