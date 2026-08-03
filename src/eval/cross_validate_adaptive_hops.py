"""Phase D: does an adaptive, per-spot-learned hop depth fix the domain-scale
mismatch diagnosed in `domain_scale_diagnostic.py`?

Leakage-safe by construction: only DLPFC held-out slices are used here (never
151673, the tuning slice, and never breast cancer/Slide-seqV2, which are the
generalization holdouts this whole investigation was trying to help). If an
architecture change can't at least hold its own on DLPFC's own held-out
slices, there is no reason to expect it will help elsewhere.

Result (3 held-out slices x 3 seeds = 9 runs per config):

    fixed_n_hops_4 (current default)          0.5038 +/- 0.0838   <- BEST
    fixed_n_hops_0 (no propagation)            0.3909 +/- 0.1143
    adaptive_hops, no regularizer              0.3501 +/- 0.1060   (collapses to depth~0)
    adaptive_hops, lambda_hop_usage=0.01       0.3351 +/- 0.1429   <- WORST, highest variance

**Verdict: REJECTED.** Without an anti-collapse regularizer, the gate
degenerates toward depth 0 for the same reason usage_entropy exists for slot
addressing: reconstruction MSE has zero incentive to use propagation, since
unsmoothed data always reconstructs more easily (measured: mean gate weight
on depth 0 > 0.999 across all 9 runs). Adding lambda_hop_usage (reusing
usage_entropy on the gate weights) successfully prevents the collapse
(effective depth rises from ~0.03 to ~2 across all 9 runs at
lambda_hop_usage=0.01), but the resulting ARI does not recover to the
fixed-n_hops=4 baseline and has the HIGHEST variance of any config tested
(0.143, vs. 0.084 for the current default) -- an initial single-slice sweep
of lambda_hop_usage in [0.001, 0.5] found even larger instability at higher
values (some seeds scored negative ARI at 0.05-0.5). Kept in the codebase as
an explicit, off-by-default ablation (`adaptive_hops=False`,
`lambda_hop_usage=0.0`) per this project's convention of preserving
tested-and-rejected mechanisms rather than deleting the evidence; see
`src/models/memory_layer.py` and `train_spatial_address.py` docstrings for
the full mechanistic account.

A second candidate explanation was checked separately and ruled out: see
`domain_scale_diagnostic.py`'s boundary-edge-weight comparison (breast
cancer's expression-weighted adjacency discriminates domain boundaries
*better* than DLPFC's, not worse).
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.models.train_spatial_address import train_spatial_address_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "adaptive_hops_check_results.json"

# Held-out slices used for this check -- deliberately excludes 151673 (the
# tuning slice for every other hyperparameter in this codebase).
CHECK_SLICES = ["151507", "151669", "151675"]
SEEDS = [0, 1, 2]

CONFIGS = {
    "fixed_n_hops_4 (current default)": dict(n_hops=4, adaptive_hops=False),
    "fixed_n_hops_0 (no propagation)": dict(n_hops=0, adaptive_hops=False),
    "adaptive_hops, no regularizer": dict(n_hops=4, adaptive_hops=True),
    # lambda_hop_usage=0.01 was the most stable value from an initial
    # single-slice sweep (0.001/0.01/0.05/0.1/0.5 tried on 151669 x 3 seeds;
    # 0.05+ showed wild instability including negative ARI on some seeds).
    "adaptive_hops, lambda_hop_usage=0.01": dict(n_hops=4, adaptive_hops=True, lambda_hop_usage=0.01),
}


def _ari(truth, labels, mask):
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def run_config(adata, truth, mask, n_layers, coords, seed, **cfg):
    _, trained, history = train_spatial_address_model(
        adata.copy(), seed=seed, verbose=False, **cfg,
    )
    labels = cluster_embedding(trained.obsm["X_spatial_address"], n_layers, coords=coords, refine=True)
    return _ari(truth, labels, mask), history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {name: [] for name in CONFIGS}
    gate_stats = []

    for sample in CHECK_SLICES:
        raw = load_dlpfc_slice(sample)
        adata = preprocess_hvg(raw.copy())
        truth = adata.obs["ground_truth_layer"]
        mask = truth.notna().to_numpy()
        n_layers = int(truth.nunique())
        coords = adata.obsm["spatial"]

        for name, cfg in CONFIGS.items():
            for seed in SEEDS:
                ari, history = run_config(adata, truth, mask, n_layers, coords, seed, device=device, **cfg)
                results[name].append(ari)
                extra = ""
                if cfg.get("adaptive_hops"):
                    eff_depth = history[-1]["hop_gate_effective_depth"]
                    gate_stats.append({"sample": sample, "seed": seed, "eff_depth": eff_depth})
                    extra = f"  eff_depth={eff_depth:.3f}"
                print(f"{sample} seed={seed} {name:38s} ARI={ari:.4f}{extra}", flush=True)

    print("\n=== SUMMARY (mean +/- std over 3 slices x 3 seeds = 9 runs) ===")
    summary = {}
    for name, aris in results.items():
        aris = np.array(aris)
        print(f"  {name:38s} {aris.mean():.4f} +/- {aris.std():.4f}")
        summary[name] = {"mean": float(aris.mean()), "std": float(aris.std()), "per_run": aris.tolist()}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(
        {"summary": summary, "gate_stats": gate_stats, "check_slices": CHECK_SLICES, "seeds": SEEDS}, indent=2
    ))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
