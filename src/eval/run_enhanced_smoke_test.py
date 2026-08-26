"""Quick 3-slice x 3-seed smoke test comparing baseline vs enhanced model variants.

Slices chosen deliberately:
  151673 -- well-studied, high-quality, prior single-slice tuning slice (included
             to anchor against known numbers; NOT used for HP selection here)
  151676 -- mid-range, held-out in Stage 8 / 11 CV
  151510 -- subject 3, highest-quality data (best library size) but persistent
             architecture-specific defect: the failure mode we most want to fix

Seeds: 0, 1, 2 (abbreviated from the full 5-seed protocol for speed)

Variants compared:
  baseline    -- original train_spatial_address_model (unchanged)
  two_stream  -- TwoStreamMemory + key repulsion + KL contrastive + entropy gate
  repulsion   -- single-stream + key repulsion only (ablate Fix 1 in isolation)
  kl_contrast -- single-stream + KL contrastive only (ablate Fix 2 in isolation)
  entropy_only-- two-stream WITHOUT entropy gate (ablate Fix 5)

Output: outputs/logs/enhanced_smoke_test_results.json
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.models.train_enhanced_model import train_enhanced_model
from src.models.train_spatial_address import train_spatial_address_model

SLICES = ["151673", "151676", "151510"]
SEEDS = [0, 1, 2]
LAYER_COL = "ground_truth_layer"
EPOCHS = 600
LOG_DIR = Path("outputs/logs")
OUT_FILE = LOG_DIR / "enhanced_smoke_test_results.json"

N_DOMAIN_SLOTS = 8
N_STATE_SLOTS = 8


def cluster_and_score(adata, embedding_key: str, n_clusters: int, seed: int) -> float:
    emb = adata.obsm[embedding_key]
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(emb)
    truth = adata.obs[LAYER_COL]
    valid = truth.notna().to_numpy()
    if valid.sum() == 0:
        return float("nan")
    truth_np = truth.to_numpy()
    return adjusted_rand_score(truth_np[valid], labels[valid])


def run_slice(sample_id: str):
    print(f"\n{'='*60}")
    print(f"  Slice: {sample_id}")
    print(f"{'='*60}")

    adata = load_dlpfc_slice(sample_id)
    adata = preprocess_hvg(adata)

    truth = adata.obs[LAYER_COL]
    n_clusters = truth[truth.notna()].nunique()
    print(f"  n_spots={adata.n_obs}  n_clusters={n_clusters}")

    results = {
        "slice": sample_id,
        "n_spots": int(adata.n_obs),
        "n_clusters": int(n_clusters),
        "variants": {},
    }

    variants = [
        # (name, kwargs to train_enhanced_model or use_baseline flag)
        ("baseline", None),
        ("repulsion_only", {
            "use_two_stream": False, "entropy_gate": False,
            "lambda_repulsion": 0.05, "lambda_kl_contrastive": 0.0,
        }),
        ("kl_contrastive_only", {
            "use_two_stream": False, "entropy_gate": False,
            "lambda_repulsion": 0.0, "lambda_kl_contrastive": 0.01,
        }),
        ("two_stream_no_entropy", {
            "use_two_stream": True, "entropy_gate": False,
            "lambda_repulsion": 0.05, "lambda_kl_contrastive": 0.01,
        }),
        ("all_fixes", {
            "use_two_stream": True, "entropy_gate": True,
            "lambda_repulsion": 0.05, "lambda_kl_contrastive": 0.01,
        }),
    ]

    for variant_name, kwargs in variants:
        aris = []
        key_sims_end = []
        t0 = time.time()
        print(f"\n  -- {variant_name} --")

        for seed in SEEDS:
            if variant_name == "baseline":
                _, adata_r, history = train_spatial_address_model(
                    adata.copy(),
                    epochs=EPOCHS,
                    seed=seed,
                    verbose=False,
                    log_every=600,
                )
                emb_key = "X_spatial_address"
                key_sim = history[-1].get("key_cosine_similarity", float("nan"))
            else:
                _, adata_r, history = train_enhanced_model(
                    adata.copy(),
                    epochs=EPOCHS,
                    seed=seed,
                    verbose=False,
                    log_every=600,
                    n_domain_slots=N_DOMAIN_SLOTS,
                    n_state_slots=N_STATE_SLOTS,
                    **kwargs,
                )
                emb_key = "X_enhanced"
                if kwargs.get("use_two_stream", True):
                    key_sim = history[-1].get("dom_key_cosine_sim", float("nan"))
                else:
                    key_sim = history[-1].get("key_cosine_sim", float("nan"))

            ari = cluster_and_score(adata_r, emb_key, n_clusters, seed)
            aris.append(ari)
            key_sims_end.append(key_sim)
            print(f"    seed={seed}  ARI={ari:.4f}  key_cos_sim={key_sim:.4f}")

        elapsed = time.time() - t0
        mean_ari = float(np.mean(aris))
        std_ari = float(np.std(aris))
        mean_cos = float(np.nanmean(key_sims_end))
        print(f"  => mean ARI={mean_ari:.4f} +/- {std_ari:.4f}  key_cos={mean_cos:.4f}  ({elapsed:.0f}s)")

        results["variants"][variant_name] = {
            "per_seed_ari": [float(a) for a in aris],
            "mean_ari": mean_ari,
            "std_ari": std_ari,
            "mean_key_cosine_sim_final": mean_cos,
            "elapsed_seconds": elapsed,
        }

    return results


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for sample_id in SLICES:
        all_results[sample_id] = run_slice(sample_id)

    # Aggregate across slices
    print(f"\n{'='*60}")
    print("  AGGREGATE SUMMARY (mean across 3 slices)")
    print(f"{'='*60}")

    variant_names = list(next(iter(all_results.values()))["variants"].keys())
    agg = {}
    for v in variant_names:
        slice_means = [all_results[s]["variants"][v]["mean_ari"] for s in SLICES]
        agg[v] = {
            "per_slice_mean_ari": dict(zip(SLICES, [round(x, 4) for x in slice_means])),
            "grand_mean_ari": float(np.mean(slice_means)),
            "grand_std_ari": float(np.std(slice_means)),
        }
        print(f"  {v:30s}  grand_mean={agg[v]['grand_mean_ari']:.4f}  "
              f"per_slice={[round(x,4) for x in slice_means]}")

    output = {
        "config": {
            "slices": SLICES,
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "n_domain_slots": N_DOMAIN_SLOTS,
            "n_state_slots": N_STATE_SLOTS,
        },
        "per_slice": all_results,
        "aggregate": agg,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
