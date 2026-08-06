"""MSAP evaluation on 10x Visium Human Breast Cancer.

The key question: does MSAP's attention pooling over hop depths allow the
model to dynamically reduce its receptive field for breast cancer's smaller
domains (28-190 spots), beating the current baseline (consensus ARI 0.546)?

Protocol:
  - Same dataset as run_breast_cancer.py (load_breast_cancer + preprocess_hvg)
  - Same 5 seeds (0-4), same clustering (mclust-equivalent + spatial refinement)
  - Same consensus protocol (cluster labels, not raw embeddings)
  - Reports mean ± std ARI across seeds, and consensus ARI
  - Reports effective hop depth per seed (key diagnostic for the generalization
    hypothesis: if attention pooling works, eff_depth should be lower than the
    baseline's fixed n_hops=4)
  - Does NOT re-run GraphST -- uses the recorded baseline (0.621 ± 0.021 per-seed,
    0.643 consensus) from outputs/logs/breast_cancer_results.json to save ~5h.

Saves to outputs/logs/msap_breast_cancer_results.json.

Comparison targets:
  Baseline (SpatialAddressMemoryLayer, DLPFC-tuned defaults):
    per-seed  0.412 ± 0.072
    consensus 0.546
  GraphST:
    per-seed  0.621 ± 0.021
    consensus 0.643
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_breast_cancer import N_REGIONS, load_breast_cancer
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.train_msap_model import train_msap_model

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "logs" / "msap_breast_cancer_results.json"
)

# Recorded baselines (from outputs/logs/breast_cancer_results.json and
# results_table.md Stage 13) -- used for printed comparison only.
_BASELINE_OURS = {"mean": 0.412, "std": 0.072, "consensus": 0.546}
_BASELINE_GRAPHST = {"mean": 0.621, "std": 0.021, "consensus": 0.643}


def _ari(truth, labels, mask):
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def evaluate(seeds, device):
    raw = load_breast_cancer()
    adata = preprocess_hvg(raw.copy(), platform="visium")

    truth = adata.obs["ground_truth_region"]
    mask = truth.notna().to_numpy()
    coords = adata.obsm["spatial"]

    results = {
        "n_spots": int(adata.n_obs),
        "n_regions": N_REGIONS,
        "model": "MSAP",
        "seeds": seeds,
        "baselines": {
            "ours_SpatialAddressMemoryLayer": _BASELINE_OURS,
            "graphst": _BASELINE_GRAPHST,
        },
    }

    msap_labels, msap_aris, seed_diagnostics = [], [], []
    for seed in seeds:
        print(f"\n  --- seed {seed} ---", flush=True)
        model, trained, history = train_msap_model(
            adata.copy(),
            seed=seed,
            device=device,
            verbose=True,
            log_every=150,
        )
        embedding = trained.obsm["X_msap"]
        labels = cluster_embedding(embedding, N_REGIONS, coords=coords, refine=True)
        msap_labels.append(labels)
        ari = _ari(truth, labels, mask)
        msap_aris.append(ari)

        final = history[-1]
        seed_diagnostics.append({
            "seed": seed,
            "ari": ari,
            "n_slots_used": final["n_slots_used"],
            "usage_entropy": final["usage_entropy"],
            "effective_hop_depth": final["effective_hop_depth"],
            "hop_weights_mean": final["hop_weights_mean"],
            "collapsed": (
                final["n_slots_used"] <= 1
                or final["usage_entropy"] < 0.5 * final["max_entropy"]
            ),
        })
        print(f"  seed {seed}: ARI={ari:.4f}  "
              f"eff_hop={final['effective_hop_depth']:.2f}  "
              f"slots_used={final['n_slots_used']}", flush=True)

    msap_consensus = consensus_cluster(msap_labels, N_REGIONS)
    consensus_ari = _ari(truth, msap_consensus, mask)

    results["msap"] = {
        "per_seed": msap_aris,
        "mean": float(np.mean(msap_aris)),
        "std": float(np.std(msap_aris)),
        "consensus": consensus_ari,
        "seed_diagnostics": seed_diagnostics,
    }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(5))

    print(f"=== MSAP Breast Cancer Evaluation ===")
    print(f"Device: {device}  |  Seeds: {seeds}")
    print(f"Baseline (SpatialAddressMemoryLayer):  "
          f"per-seed={_BASELINE_OURS['mean']:.3f}  "
          f"consensus={_BASELINE_OURS['consensus']:.3f}")
    print(f"GraphST baseline:                      "
          f"per-seed={_BASELINE_GRAPHST['mean']:.3f}  "
          f"consensus={_BASELINE_GRAPHST['consensus']:.3f}")

    results = evaluate(seeds, device)

    msap = results["msap"]
    print(f"\n=== Final Results ===")
    print(f"  MSAP:             per-seed = {msap['mean']:.4f} ± {msap['std']:.4f}  "
          f"consensus = {msap['consensus']:.4f}")
    print(f"  Baseline (ours):  per-seed = {_BASELINE_OURS['mean']:.4f} ± "
          f"{_BASELINE_OURS['std']:.4f}  "
          f"consensus = {_BASELINE_OURS['consensus']:.4f}")
    print(f"  GraphST:          per-seed = {_BASELINE_GRAPHST['mean']:.4f} ± "
          f"{_BASELINE_GRAPHST['std']:.4f}  "
          f"consensus = {_BASELINE_GRAPHST['consensus']:.4f}")
    print(f"\n  delta consensus (MSAP - baseline): "
          f"{msap['consensus'] - _BASELINE_OURS['consensus']:+.4f}")
    print(f"  delta consensus (MSAP - GraphST):  "
          f"{msap['consensus'] - _BASELINE_GRAPHST['consensus']:+.4f}")

    print(f"\n  Per-seed hop diagnostics:")
    for d in msap["seed_diagnostics"]:
        hops = [f"{w:.3f}" for w in d["hop_weights_mean"]]
        print(f"    seed {d['seed']}: ARI={d['ari']:.4f}  "
              f"eff_hop={d['effective_hop_depth']:.2f}  "
              f"slots={d['n_slots_used']}  collapsed={d['collapsed']}")
        print(f"      hop_weights: [{', '.join(hops)}]")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
