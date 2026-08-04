"""Run Hop-Fusion on every real dataset used by the Phase 0/C benchmarks.

This is an evaluation runner, not a tuning script.  The default configuration
is explicitly recorded as provisional until the leakage-safe DLPFC selector
has produced a locked config.  It applies the same physical radius to every
platform and reports the platform-specific measured edge length and resulting
hop count for every dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS, load_breast_cancer
from src.data.load_dlpfc import ALL_DLPFC_SAMPLES, load_dlpfc_slice
from src.data.load_slideseqv2 import load_slideseqv2
from src.data.load_visium import load_visium_crop, load_visium_full
from src.data.preprocess import preprocess_hvg
from src.data.physical_scale import get_average_edge_length_um
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.eval.metrics import embedding_silhouette, spatial_coherence
from src.models.train_hop_fusion import train_hop_fusion_model


OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "hop_fusion_legacy_results.json"
SLIDE_SUBSAMPLE_N = 12000
SLIDE_SUBSAMPLE_SEED = 0
VISIUM_CLUSTER_COUNTS = {"crop": 14, "full": 25}


def _train(adata, platform, physical_radius_um, seed, epochs, device, config):
    _, trained, history = train_hop_fusion_model(
        adata.copy(),
        platform=platform,
        physical_radius_um=physical_radius_um,
        seed=seed,
        epochs=epochs,
        device=device,
        verbose=False,
        **config,
    )
    return trained, history


def _annotated_result(adata, truth_key, n_clusters, seeds, platform, radius, epochs, device, config):
    truth = adata.obs[truth_key].to_numpy()
    valid = adata.obs[truth_key].notna().to_numpy()
    labels_by_seed = []
    aris = []
    metadata = None
    for seed in seeds:
        trained, history = _train(adata, platform, radius, seed, epochs, device, config)
        labels = cluster_embedding(
            trained.obsm["X_hop_fusion"], n_clusters,
            coords=adata.obsm["spatial"], refine=True,
        )
        labels_by_seed.append(labels)
        aris.append(float(adjusted_rand_score(truth[valid], labels[valid])))
        metadata = trained.uns["hop_fusion"]
    result = {
        "n_spots": int(adata.n_obs),
        "n_clusters": int(n_clusters),
        "per_seed_ari": aris,
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris)),
        "consensus_ari": float(
            adjusted_rand_score(truth[valid], consensus_cluster(labels_by_seed, n_clusters)[valid])
        ),
        "physical_metadata": metadata,
    }
    return result


def _unsupervised_result(adata, n_clusters, seeds, platform, radius, epochs, device, config):
    rows = []
    metadata = None
    for seed in seeds:
        trained, history = _train(adata, platform, radius, seed, epochs, device, config)
        labels = cluster_embedding(
            trained.obsm["X_hop_fusion"], n_clusters,
            coords=adata.obsm["spatial"], refine=True,
        )
        trained.obs["_hop_fusion_pred"] = labels
        rows.append({
            "seed": seed,
            "silhouette": embedding_silhouette(trained.obsm["X_hop_fusion"], labels),
            "spatial_coherence_morans_i": spatial_coherence(
                trained, "_hop_fusion_pred"
            )["mean"],
        })
        metadata = trained.uns["hop_fusion"]
    return {
        "n_spots": int(adata.n_obs),
        "n_clusters": int(n_clusters),
        "per_seed": rows,
        "silhouette_mean": float(np.mean([row["silhouette"] for row in rows])),
        "silhouette_std": float(np.std([row["silhouette"] for row in rows])),
        "spatial_coherence_mean": float(
            np.mean([row["spatial_coherence_morans_i"] for row in rows])
        ),
        "spatial_coherence_std": float(
            np.std([row["spatial_coherence_morans_i"] for row in rows])
        ),
        "physical_metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--dlpfc-seeds", type=int, default=5)
    parser.add_argument("--breast-seeds", type=int, default=5)
    parser.add_argument("--slide-seeds", type=int, default=3)
    parser.add_argument("--visium-seeds", type=int, default=1)
    parser.add_argument("--physical-radius-um", type=float, default=220.0)
    parser.add_argument("--skip-dlpfc", action="store_true")
    parser.add_argument("--skip-breast", action="store_true")
    parser.add_argument("--skip-slide", action="store_true")
    parser.add_argument("--skip-visium", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Current DLPFC-validated defaults from the existing fixed-hop model;
    # lambda_spatial_coherence stays zero so this is concat-fusion alone.
    config = {
        "memory_slots": 16,
        "memory_dim": 128,
        "hidden_dim": 256,
        "fusion_hidden_dim": 128,
        "fusion_depth": 2,
        "temperature": 1.0,
        "attention_fn": "softmax",
        "lambda_usage": 0.02,
        "lambda_spatial_coherence": 0.0,
        "expression_weighted": True,
    }
    results = {
        "status": "provisional_concat_fusion_real_data_run",
        "configuration": {
            **config,
            "physical_radius_um": args.physical_radius_um,
            "epochs": args.epochs,
            "device": str(device),
            "dlpfc_selection_status": "not_locked; radius is current fixed-hop reference 4 * 55 um",
        },
        "datasets": {},
    }

    if not args.skip_dlpfc:
        results["datasets"]["dlpfc"] = {"per_slice": {}}
        for sample in ALL_DLPFC_SAMPLES:
            adata = preprocess_hvg(load_dlpfc_slice(sample), platform="visium")
            results["datasets"]["dlpfc"]["per_slice"][sample] = _annotated_result(
                adata, "ground_truth_layer", int(adata.obs["ground_truth_layer"].nunique()),
                list(range(args.dlpfc_seeds)), "visium", args.physical_radius_um,
                args.epochs, device, config,
            )
            row = results["datasets"]["dlpfc"]["per_slice"][sample]
            print(
                f"DLPFC {sample}: ARI={row['mean_ari']:.4f} +/- {row['std_ari']:.4f} "
                f"consensus={row['consensus_ari']:.4f} hops={row['physical_metadata']['fusion_hops']}",
                flush=True,
            )
        per_slice = results["datasets"]["dlpfc"]["per_slice"]
        held_out = [row for sample, row in per_slice.items() if sample != "151673"]
        results["datasets"]["dlpfc"]["summary"] = {
            "held_out_11_mean_ari": float(np.mean([row["mean_ari"] for row in held_out])),
            "held_out_11_std_ari": float(np.std([row["mean_ari"] for row in held_out])),
            "held_out_11_mean_consensus_ari": float(np.mean([row["consensus_ari"] for row in held_out])),
            "all_12_mean_ari": float(np.mean([row["mean_ari"] for row in per_slice.values()])),
            "all_12_mean_consensus_ari": float(np.mean([row["consensus_ari"] for row in per_slice.values()])),
        }

    if not args.skip_breast:
        adata = preprocess_hvg(load_breast_cancer().copy(), platform="visium")
        results["datasets"]["breast_cancer"] = _annotated_result(
            adata, "ground_truth_region", N_REGIONS, list(range(args.breast_seeds)),
            "visium", args.physical_radius_um, args.epochs, device, config,
        )
        print("Breast cancer:", results["datasets"]["breast_cancer"], flush=True)

    if not args.skip_slide:
        raw = load_slideseqv2()
        rng = np.random.default_rng(SLIDE_SUBSAMPLE_SEED)
        keep = rng.choice(raw.n_obs, size=min(SLIDE_SUBSAMPLE_N, raw.n_obs), replace=False)
        adata = preprocess_hvg(raw[np.sort(keep)].copy(), coord_type="generic", platform="slideseqv2")
        results["datasets"]["slideseqv2"] = _unsupervised_result(
            adata, 14, list(range(args.slide_seeds)), "slideseqv2",
            args.physical_radius_um, args.epochs, device, config,
        )
        results["datasets"]["slideseqv2"]["subsampled_from"] = int(raw.n_obs)
        print("Slide-seqV2:", results["datasets"]["slideseqv2"], flush=True)

    if not args.skip_visium:
        results["datasets"]["visium"] = {}
        for name, loader in (("crop", load_visium_crop), ("full", load_visium_full)):
            adata = preprocess_hvg(loader(), platform="visium")
            results["datasets"]["visium"][name] = _unsupervised_result(
                adata, VISIUM_CLUSTER_COUNTS[name], list(range(args.visium_seeds)),
                "visium", args.physical_radius_um, args.epochs, device, config,
            )
            print(f"Visium {name}:", results["datasets"]["visium"][name], flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
