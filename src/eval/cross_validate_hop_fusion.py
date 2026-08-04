"""Leakage-safe DLPFC selection for Hop-Fusion.

The selector uses the same three-slice validation partition as the existing
capacity/hop cross-validation.  DLPFC 151673 remains excluded, and the eight
true holdout slices are not used to choose anything.  Once selection is done,
the resulting physical radius, fusion MLP shape, and coherence-loss weight are
written to ``configs/hop_fusion_dlpfc.json`` for all downstream datasets.

The script deliberately reports three separate mechanisms:

1. concat-fusion alone;
2. concat-fusion plus the new address-coherence loss;
3. the loss on the existing fixed-hop model, without concat-fusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.physical_scale import get_average_edge_length_um
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.eval.cross_validate_capacity import CV_VALIDATION_SLICES, TRUE_HOLDOUT_SLICES
from src.eval.hop_fusion_protocol import DEFAULT_LOCK_PATH, DEFAULT_SELECTION_PATH, load_json
from src.models.train_hop_fusion import train_hop_fusion_model
from src.models.train_spatial_address import train_spatial_address_model


def _ari(adata, embedding_key):
    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    n_layers = int(truth.nunique())
    labels = cluster_embedding(
        adata.obsm[embedding_key], n_layers, coords=adata.obsm["spatial"], refine=True
    )
    return float(adjusted_rand_score(truth.to_numpy()[mask], np.asarray(labels)[mask]))


def _fusion_score(adata, config, seed, device, epochs):
    _, trained, _ = train_hop_fusion_model(
        adata.copy(),
        platform="visium",
        physical_radius_um=config["physical_radius_um"],
        memory_slots=config["memory_slots"],
        memory_dim=config["memory_dim"],
        hidden_dim=config["hidden_dim"],
        fusion_hidden_dim=config["fusion_hidden_dim"],
        fusion_depth=config["fusion_depth"],
        lambda_usage=config["lambda_usage"],
        lambda_spatial_coherence=config["lambda_spatial_coherence"],
        expression_weighted=config["expression_weighted"],
        attention_fn=config["attention_fn"],
        seed=seed,
        epochs=epochs,
        device=device,
        verbose=False,
    )
    return _ari(trained, "X_hop_fusion")


def _mean_cv_score(cache, config, seeds, device, epochs):
    per_slice = []
    for sample, adata in cache.items():
        scores = [_fusion_score(adata, config, seed, device, epochs) for seed in seeds]
        per_slice.append(float(np.mean(scores)))
    return float(np.mean(per_slice)), per_slice


def _select(candidates, base_config, field, cache, seeds, device, epochs):
    scores = {}
    for value in candidates:
        config = dict(base_config)
        config[field] = value
        if field == "reference_max_hops":
            config["physical_radius_um"] = value * config["reference_edge_length_um"]
        score, per_slice = _mean_cv_score(cache, config, seeds, device, epochs)
        scores[str(value)] = {"mean": score, "per_slice": per_slice}
        print(f"{field}={value}: CV mean={score:.4f}", flush=True)
    selected = max(candidates, key=lambda value: scores[str(value)]["mean"])
    print(f"selected {field}={selected}\n", flush=True)
    return selected, scores


def _ablation_scores(cache, config, seeds, device, epochs):
    results = {}
    concat_only = dict(config, lambda_spatial_coherence=0.0)
    results["concat_fusion_alone"] = _mean_cv_score(
        cache, concat_only, seeds, device, epochs
    )[0]
    results["concat_fusion_plus_spatial_coherence"] = _mean_cv_score(
        cache, config, seeds, device, epochs
    )[0]

    fixed_scores = []
    for adata in cache.values():
        per_seed = []
        for seed in seeds:
            _, trained, _ = train_spatial_address_model(
                adata.copy(),
                n_hops=config["reference_max_hops"],
                memory_slots=config["memory_slots"],
                memory_dim=config["memory_dim"],
                hidden_dim=config["hidden_dim"],
                lambda_usage=config["lambda_usage"],
                lambda_spatial_coherence=config["lambda_spatial_coherence"],
                expression_weighted=config["expression_weighted"],
                attention_fn=config["attention_fn"],
                seed=seed,
                epochs=epochs,
                device=device,
                verbose=False,
            )
            per_seed.append(_ari(trained, "X_spatial_address"))
        fixed_scores.append(float(np.mean(per_seed)))
    results["spatial_coherence_loss_alone"] = float(np.mean(fixed_scores))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-true-holdout", action="store_true")
    args = parser.parse_args()

    selection = load_json(DEFAULT_SELECTION_PATH)
    fixed = selection["fixed_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {
        sample: preprocess_hvg(load_dlpfc_slice(sample), platform="visium")
        for sample in selection["cv_validation_slices"]
    }
    reference_edge_length_um = float(
        np.median([get_average_edge_length_um(adata, "visium") for adata in cache.values()])
    )
    print(f"Measured CV reference edge length: {reference_edge_length_um:.4f} um")

    base = {
        **fixed,
        "reference_max_hops": selection["candidates"]["reference_max_hops"][0],
        "reference_edge_length_um": reference_edge_length_um,
        "physical_radius_um": reference_edge_length_um,
        "fusion_hidden_dim": selection["candidates"]["fusion_hidden_dim"][0],
        "fusion_depth": selection["candidates"]["fusion_depth"][0],
        "lambda_spatial_coherence": 0.0,
    }
    selected_hops, hop_scores = _select(
        selection["candidates"]["reference_max_hops"],
        base,
        "reference_max_hops",
        cache,
        args.seeds,
        device,
        args.epochs,
    )
    base["reference_max_hops"] = selected_hops
    base["physical_radius_um"] = selected_hops * reference_edge_length_um
    selected_width, width_scores = _select(
        selection["candidates"]["fusion_hidden_dim"],
        base,
        "fusion_hidden_dim",
        cache,
        args.seeds,
        device,
        args.epochs,
    )
    base["fusion_hidden_dim"] = selected_width
    selected_depth, depth_scores = _select(
        selection["candidates"]["fusion_depth"],
        base,
        "fusion_depth",
        cache,
        args.seeds,
        device,
        args.epochs,
    )
    base["fusion_depth"] = selected_depth
    selected_lambda, lambda_scores = _select(
        selection["candidates"]["lambda_spatial_coherence"],
        base,
        "lambda_spatial_coherence",
        cache,
        args.seeds,
        device,
        args.epochs,
    )
    base["lambda_spatial_coherence"] = selected_lambda
    base["status"] = "locked"
    base["selection_protocol"] = {
        "cv_validation_slices": selection["cv_validation_slices"],
        "true_holdout_slices": selection["true_holdout_slices"],
        "tuning_slice_excluded": selection["tuning_slice_excluded"],
        "seeds": args.seeds,
        "epochs": args.epochs,
        "reference_edge_length_um": reference_edge_length_um,
    }
    base["reference_platform"] = selection["reference_platform"]
    base["fixed_baseline_n_hops"] = selection["fixed_baseline_n_hops"]

    ablations = _ablation_scores(cache, base, args.seeds, device, args.epochs)
    base["ablation_cv_scores"] = ablations
    base["selection_scores"] = {
        "reference_max_hops": hop_scores,
        "fusion_hidden_dim": width_scores,
        "fusion_depth": depth_scores,
        "lambda_spatial_coherence": lambda_scores,
    }
    DEFAULT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOCK_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"\nWrote locked config to {DEFAULT_LOCK_PATH}")
    print("DLPFC CV ablations:", json.dumps(ablations, indent=2))

    if not args.skip_true_holdout:
        holdout_cache = {
            sample: preprocess_hvg(load_dlpfc_slice(sample), platform="visium")
            for sample in selection["true_holdout_slices"]
        }
        mean, per_slice = _mean_cv_score(
            holdout_cache, base, args.seeds, device, args.epochs
        )
        result = {
            "config": str(DEFAULT_LOCK_PATH),
            "true_holdout_mean": mean,
            "true_holdout_per_slice": dict(zip(holdout_cache, per_slice)),
        }
        output = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "hop_fusion_dlpfc_selection.json"
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote true-holdout result to {output}")


if __name__ == "__main__":
    main()
