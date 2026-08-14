"""Standard protocol evaluation for LDCM architecture.

This follows the master rerun plan's standardized protocol:
- DLPFC: 3 selection slices + 8 report slices, 5 seeds each
- Breast cancer: 2 selection blocks + 4 report blocks, 5 seeds each
- Identical clustering protocol for all methods
- Incremental JSON logging for crash recovery
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import ALL_DLPFC_SAMPLES, load_dlpfc_slice
from src.data.load_breast_cancer import load_breast_cancer
from src.data.preprocess import preprocess_hvg
from src.eval.breast_cancer_spatial_blocks import (
    load_or_build_blocks,
    get_selection_mask,
    get_report_mask,
    SELECTION_BLOCKS,
    REPORT_BLOCKS,
)
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.train_ldcm_model import train_ldcm_model, consensus_from_embeddings

# Standard hyperparameters (locked project-wide)
BASELINE_HYPERPARAMS = {
    "memory_slots": 16,
    "memory_dim": 128,
    "hidden_dim": 256,
    "n_hops": 4,
    "temperature": 1.0,
    "lambda_usage": 0.02,
    "expression_weighted": True,
}

# LDCM-specific hyperparameter (to be selected on selection blocks/slices)
CONTRASTIVE_GRID = [0.05, 0.1, 0.2, 0.5]  # lambda_contrastive values to test

# Output paths
DLPFC_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "ldcm_dlpfc_results.json"
BREAST_CANCER_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "ldcm_breast_cancer_results.json"

# DLPFC selection slices (existing selection subset, 151673 excluded)
DLPFC_SELECTION_SLICES = ["151507", "151669", "151670"]
DLPFC_REPORT_SLICES = [s for s in ALL_DLPFC_SAMPLES if s not in DLPFC_SELECTION_SLICES and s != "151673"]


def _ari(truth, labels, mask):
    """Compute ARI for masked spots."""
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def _save_incremental_result(output_path, results):
    """Save results incrementally for crash recovery."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def _select_lambda_contrastive_breast_cancer():
    """Select lambda_contrastive on breast cancer selection blocks."""
    print("\n" + "="*60)
    print("LDCM Breast Cancer: Hyperparameter Selection")
    print("="*60)
    
    adata = load_breast_cancer()
    preprocess_hvg(adata)
    block_id = load_or_build_blocks()
    selection_mask = get_selection_mask(block_id)
    
    best_lambda = None
    best_score = -np.inf
    selection_results = {}
    
    for lambda_c in CONTRASTIVE_GRID:
        print(f"\nTesting lambda_contrastive={lambda_c}")
        
        seed_scores = []
        for seed in range(3):  # Use fewer seeds for selection to save time
            print(f"  Seed {seed}...", end=" ")
            
            model, adata_trained, history = train_ldcm_model(
                adata,
                epochs=400,  # Fewer epochs for selection
                lambda_contrastive=lambda_c,
                seed=seed,
                verbose=False,
                **BASELINE_HYPERPARAMS
            )
            
            embedding = adata_trained.obsm["X_ldcm"]
            n_clusters = adata.obs["ground_truth_region"].nunique()
            labels = cluster_embedding(embedding, n_clusters)
            score = _ari(adata.obs["ground_truth_region"].values, labels, selection_mask)
            seed_scores.append(score)
            print(f"ARI={score:.4f}")
        
        mean_score = np.mean(seed_scores)
        selection_results[str(lambda_c)] = {
            "mean_ari": float(mean_score),
            "seed_aris": [float(s) for s in seed_scores],
        }
        
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lambda_c
    
    print(f"\nBest lambda_contrastive: {best_lambda} (selection ARI={best_score:.4f})")
    return best_lambda, selection_results


def _select_lambda_contrastive_dlpfc():
    """Select lambda_contrastive on DLPFC selection slices."""
    print("\n" + "="*60)
    print("LDCM DLPFC: Hyperparameter Selection")
    print("="*60)
    
    best_lambda = None
    best_score = -np.inf
    selection_results = {}
    
    for lambda_c in CONTRASTIVE_GRID:
        print(f"\nTesting lambda_contrastive={lambda_c}")
        
        slice_scores = []
        for slice_id in DLPFC_SELECTION_SLICES:
            print(f"  Slice {slice_id}...", end=" ")
            
            adata = load_dlpfc_slice(slice_id)
            preprocess_hvg(adata)
            
            # Use single seed for selection to save time
            model, adata_trained, history = train_ldcm_model(
                adata,
                epochs=400,
                lambda_contrastive=lambda_c,
                seed=0,
                verbose=False,
                **BASELINE_HYPERPARAMS
            )
            
            embedding = adata_trained.obsm["X_ldcm"]
            n_clusters = adata.obs["ground_truth_layer"].nunique()
            labels = cluster_embedding(embedding, n_clusters)
            score = _ari(adata.obs["ground_truth_layer"].values, labels, np.ones(len(labels), dtype=bool))
            slice_scores.append(score)
            print(f"ARI={score:.4f}")
        
        mean_score = np.mean(slice_scores)
        selection_results[str(lambda_c)] = {
            "mean_ari": float(mean_score),
            "slice_aris": [float(s) for s in slice_scores],
        }
        
        if mean_score > best_score:
            best_score = mean_score
            best_lambda = lambda_c
    
    print(f"\nBest lambda_contrastive: {best_lambda} (selection ARI={best_score:.4f})")
    return best_lambda, selection_results


def evaluate_breast_cancer():
    """Full evaluation on breast cancer with block protocol."""
    print("\n" + "="*60)
    print("LDCM Breast Cancer: Full Evaluation")
    print("="*60)
    
    # Hyperparameter selection
    best_lambda, selection_results = _select_lambda_contrastive_breast_cancer()
    
    # Load data and blocks
    adata = load_breast_cancer()
    preprocess_hvg(adata)
    block_id = load_or_build_blocks()
    report_mask = get_report_mask(block_id)
    n_clusters = adata.obs["ground_truth_region"].nunique()
    
    # Initialize results structure
    results = {
        "architecture": "LDCM",
        "dataset": "breast_cancer",
        "hyperparameters": {**BASELINE_HYPERPARAMS, "lambda_contrastive": best_lambda},
        "selection": selection_results,
        "report": {
            "blocks": {},
            "per_seed_mean": None,
            "consensus": None,
        },
    }
    
    # Evaluation on report blocks
    print(f"\nEvaluating on {len(REPORT_BLOCKS)} report blocks with lambda_contrastive={best_lambda}")
    
    seed_embeddings = []
    for seed in range(5):
        print(f"\nSeed {seed}:")
        block_aris = []
        
        for block_id in REPORT_BLOCKS:
            block_mask = (block_id == block_id)
            full_mask = report_mask & block_mask
            
            print(f"  Block {block_id}...", end=" ")
            
            model, adata_trained, history = train_ldcm_model(
                adata,
                epochs=600,
                lambda_contrastive=best_lambda,
                seed=seed,
                verbose=False,
                **BASELINE_HYPERPARAMS
            )
            
            embedding = adata_trained.obsm["X_ldcm"]
            labels = cluster_embedding(embedding, n_clusters)
            ari = _ari(adata.obs["ground_truth_region"].values, labels, full_mask)
            block_aris.append(ari)
            print(f"ARI={ari:.4f}")
            
            # Store per-block result
            if str(block_id) not in results["report"]["blocks"]:
                results["report"]["blocks"][str(block_id)] = {}
            results["report"]["blocks"][str(block_id)][f"seed_{seed}"] = float(ari)
        
        seed_embeddings.append(adata_trained.obsm["X_ldcm"])
        mean_ari = np.mean(block_aris)
        print(f"  Seed {seed} mean ARI: {mean_ari:.4f}")
        
        # Save incremental results
        _save_incremental_result(BREAST_CANCER_OUTPUT_PATH, results)
    
    # Compute consensus
    print("\nComputing consensus clustering...")
    consensus_labels = consensus_from_embeddings(seed_embeddings, n_clusters)
    consensus_ari = _ari(adata.obs["ground_truth_region"].values, consensus_labels, report_mask)
    results["report"]["consensus"] = float(consensus_ari)
    
    # Compute per-seed mean
    per_seed_means = []
    for block_id in REPORT_BLOCKS:
        block_seed_scores = [results["report"]["blocks"][str(block_id)][f"seed_{seed}"] for seed in range(5)]
        per_seed_means.append(np.mean(block_seed_scores))
    results["report"]["per_seed_mean"] = float(np.mean(per_seed_means))
    results["report"]["per_seed_std"] = float(np.std(per_seed_means))
    
    print(f"\nBreast cancer results:")
    print(f"  Per-seed mean: {results['report']['per_seed_mean']:.4f} ± {results['report']['per_seed_std']:.4f}")
    print(f"  Consensus: {results['report']['consensus']:.4f}")
    
    # Final save
    _save_incremental_result(BREAST_CANCER_OUTPUT_PATH, results)
    
    # Verify output
    with open(BREAST_CANCER_OUTPUT_PATH, "r") as f:
        loaded = json.load(f)
    assert loaded["report"]["consensus"] is not None, "Consensus result missing"
    print(f"\nResults saved to {BREAST_CANCER_OUTPUT_PATH}")
    
    return results


def evaluate_dlpfc():
    """Full evaluation on DLPFC with slice protocol."""
    print("\n" + "="*60)
    print("LDCM DLPFC: Full Evaluation")
    print("="*60)
    
    # Hyperparameter selection
    best_lambda, selection_results = _select_lambda_contrastive_dlpfc()
    
    # Initialize results structure
    results = {
        "architecture": "LDCM",
        "dataset": "dlpfc",
        "hyperparameters": {**BASELINE_HYPERPARAMS, "lambda_contrastive": best_lambda},
        "selection": selection_results,
        "report": {
            "slices": {},
            "per_seed_mean": None,
            "consensus": None,
        },
    }
    
    # Evaluation on report slices
    print(f"\nEvaluating on {len(DLPFC_REPORT_SLICES)} report slices with lambda_contrastive={best_lambda}")
    
    seed_embeddings = {slice_id: [] for slice_id in DLPFC_REPORT_SLICES}
    
    for seed in range(5):
        print(f"\nSeed {seed}:")
        
        for slice_id in DLPFC_REPORT_SLICES:
            print(f"  Slice {slice_id}...", end=" ")
            
            adata = load_dlpfc_slice(slice_id)
            preprocess_hvg(adata)
            n_clusters = adata.obs["ground_truth_layer"].nunique()
            
            model, adata_trained, history = train_ldcm_model(
                adata,
                epochs=600,
                lambda_contrastive=best_lambda,
                seed=seed,
                verbose=False,
                **BASELINE_HYPERPARAMS
            )
            
            embedding = adata_trained.obsm["X_ldcm"]
            labels = cluster_embedding(embedding, n_clusters)
            ari = _ari(adata.obs["ground_truth_layer"].values, labels, np.ones(len(labels), dtype=bool))
            
            seed_embeddings[slice_id].append(embedding)
            
            # Store per-slice result
            if slice_id not in results["report"]["slices"]:
                results["report"]["slices"][slice_id] = {}
            results["report"]["slices"][slice_id][f"seed_{seed}"] = float(ari)
            print(f"ARI={ari:.4f}")
        
        # Save incremental results
        _save_incremental_result(DLPFC_OUTPUT_PATH, results)
    
    # Compute consensus per slice
    print("\nComputing consensus clustering per slice...")
    consensus_aris = []
    for slice_id in DLPFC_REPORT_SLICES:
        slice_embeddings = seed_embeddings[slice_id]
        adata = load_dlpfc_slice(slice_id)
        n_clusters = adata.obs["ground_truth_layer"].nunique()
        
        consensus_labels = consensus_from_embeddings(slice_embeddings, n_clusters)
        consensus_ari = _ari(adata.obs["ground_truth_layer"].values, consensus_labels, np.ones(len(consensus_labels), dtype=bool))
        results["report"]["slices"][slice_id]["consensus"] = float(consensus_ari)
        consensus_aris.append(consensus_ari)
    
    results["report"]["consensus"] = float(np.mean(consensus_aris))
    results["report"]["consensus_std"] = float(np.std(consensus_aris))
    
    # Compute per-seed mean
    per_seed_means = []
    for seed in range(5):
        seed_slice_scores = [results["report"]["slices"][slice_id][f"seed_{seed}"] for slice_id in DLPFC_REPORT_SLICES]
        per_seed_means.append(np.mean(seed_slice_scores))
    results["report"]["per_seed_mean"] = float(np.mean(per_seed_means))
    results["report"]["per_seed_std"] = float(np.std(per_seed_means))
    
    print(f"\nDLPFC results:")
    print(f"  Per-seed mean: {results['report']['per_seed_mean']:.4f} ± {results['report']['per_seed_std']:.4f}")
    print(f"  Consensus: {results['report']['consensus']:.4f} ± {results['report']['consensus_std']:.4f}")
    
    # Final save
    _save_incremental_result(DLPFC_OUTPUT_PATH, results)
    
    # Verify output
    with open(DLPFC_OUTPUT_PATH, "r") as f:
        loaded = json.load(f)
    assert loaded["report"]["consensus"] is not None, "Consensus result missing"
    print(f"\nResults saved to {DLPFC_OUTPUT_PATH}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LDCM standard protocol evaluation")
    parser.add_argument("--dataset", choices=["breast_cancer", "dlpfc", "both"], default="both",
                       help="Which dataset to evaluate")
    args = parser.parse_args()
    
    if args.dataset in ["breast_cancer", "both"]:
        evaluate_breast_cancer()
    
    if args.dataset in ["dlpfc", "both"]:
        evaluate_dlpfc()
    
    print("\n" + "="*60)
    print("LDCM evaluation complete")
    print("="*60)