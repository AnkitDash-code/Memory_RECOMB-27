"""PPR (Personalized PageRank address propagation) evaluation on 10x Visium
Human Breast Cancer.

An earlier informal session tested PPR with alpha=0.2 hardcoded (never
selected) and got a single 5-seed consensus of 0.527, below baseline
(0.546) -- but that run's log file was never persisted, so it isn't a
verified result. This script fixes that: a light alpha pre-sweep (3
candidates x 2 seeds, cheap) picks a value before the full 5-seed
evaluation, and everything is actually saved this time.

Caveat, stated honestly: like BAAP/HMA/MSAP/GMSM/AGAP, this reuses the
whole breast-cancer sample for both alpha selection and final scoring --
it is NOT a nested spatial-block holdout (unlike LDCM's standardized
protocol). Given PPR is expected to underperform based on the informal
prior test, this is an acceptable first closure pass; only worth a proper
block-holdout rerun if this pass looks unexpectedly promising.

Saves output to outputs/logs/ppr_breast_cancer_results.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS, load_breast_cancer
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.train_ppr_model import train_ppr_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "ppr_breast_cancer_results.json"

_BASELINE_OURS = {"mean": 0.412, "std": 0.072, "consensus": 0.546}
_BASELINE_GRAPHST = {"mean": 0.621, "std": 0.021, "consensus": 0.643}
ALPHA_GRID = [0.1, 0.2, 0.4]


def _ari(truth, labels, mask):
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def _select_alpha(adata, truth, mask, coords, device):
    print("\n--- alpha pre-sweep (2 seeds each, whole-sample scoring) ---")
    best_alpha, best_score, sweep = None, -np.inf, {}
    for alpha in ALPHA_GRID:
        seed_scores = []
        for seed in (0, 1):
            _, trained, _ = train_ppr_model(
                adata.copy(), alpha=alpha, seed=seed, device=device, verbose=False, log_every=600,
            )
            labels = cluster_embedding(trained.obsm["X_ppr"], N_REGIONS, coords=coords, refine=True)
            seed_scores.append(_ari(truth, labels, mask))
        mean_score = float(np.mean(seed_scores))
        sweep[str(alpha)] = {"mean_ari": mean_score, "seed_aris": seed_scores}
        print(f"  alpha={alpha}: mean ARI={mean_score:.4f}  {seed_scores}")
        if mean_score > best_score:
            best_score, best_alpha = mean_score, alpha
    print(f"  selected alpha={best_alpha} (pre-sweep ARI={best_score:.4f})")
    return best_alpha, sweep


def evaluate(seeds, device):
    raw = load_breast_cancer()
    adata = preprocess_hvg(raw.copy(), platform="visium")

    truth = adata.obs["ground_truth_region"]
    mask = truth.notna().to_numpy()
    coords = adata.obsm["spatial"]

    best_alpha, alpha_sweep = _select_alpha(adata, truth, mask, coords, device)

    results = {
        "n_spots": int(adata.n_obs),
        "n_regions": N_REGIONS,
        "model": "PPR",
        "seeds": seeds,
        "alpha_sweep": alpha_sweep,
        "selected_alpha": best_alpha,
        "baselines": {
            "ours_SpatialAddressMemoryLayer": _BASELINE_OURS,
            "graphst": _BASELINE_GRAPHST,
        },
    }

    labels_per_seed, aris, seed_diagnostics = [], [], []
    for seed in seeds:
        print(f"\n  --- seed {seed} (alpha={best_alpha}) ---", flush=True)
        model, trained, history = train_ppr_model(
            adata.copy(), alpha=best_alpha, seed=seed, device=device, verbose=True, log_every=150,
        )
        embedding = trained.obsm["X_ppr"]
        labels = cluster_embedding(embedding, N_REGIONS, coords=coords, refine=True)
        labels_per_seed.append(labels)
        ari = _ari(truth, labels, mask)
        aris.append(ari)

        final = history[-1]
        collapsed = (
            final["n_slots_used"] <= 1
            or final["usage_entropy"] < 0.5 * final["max_entropy"]
        )
        seed_diagnostics.append({
            "seed": seed,
            "ari": ari,
            "n_slots_used": final["n_slots_used"],
            "usage_entropy": final["usage_entropy"],
            "collapsed": collapsed,
        })
        print(
            f"  seed {seed}: ARI={ari:.4f}  slots_used={final['n_slots_used']}  "
            f"usage_entropy={final['usage_entropy']:.3f}/{final['max_entropy']:.3f}",
            flush=True,
        )

    consensus_labels = consensus_cluster(labels_per_seed, N_REGIONS)
    consensus_ari = _ari(truth, consensus_labels, mask)

    results["ppr"] = {
        "per_seed": aris,
        "mean": float(np.mean(aris)),
        "std": float(np.std(aris)),
        "consensus": consensus_ari,
        "seed_diagnostics": seed_diagnostics,
    }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(5))

    print("=== PPR Breast Cancer Evaluation ===")
    print(f"Device: {device}  |  Seeds: {seeds}")

    results = evaluate(seeds, device)

    ppr = results["ppr"]
    print("\n=== Final Results ===")
    print(f"  PPR (alpha={results['selected_alpha']}):  per-seed = {ppr['mean']:.4f} +/- {ppr['std']:.4f}  consensus = {ppr['consensus']:.4f}")
    print(f"  Baseline (ours):  per-seed = {_BASELINE_OURS['mean']:.4f} +/- {_BASELINE_OURS['std']:.4f}  consensus = {_BASELINE_OURS['consensus']:.4f}")
    print(f"  GraphST:          per-seed = {_BASELINE_GRAPHST['mean']:.4f} +/- {_BASELINE_GRAPHST['std']:.4f}  consensus = {_BASELINE_GRAPHST['consensus']:.4f}")
    print(f"\n  delta consensus (PPR - baseline): {ppr['consensus'] - _BASELINE_OURS['consensus']:+.4f}")
    print(f"  delta consensus (PPR - GraphST):  {ppr['consensus'] - _BASELINE_GRAPHST['consensus']:+.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
