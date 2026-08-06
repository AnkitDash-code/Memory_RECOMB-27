"""MSAP smoke test: single slice (151673), single seed.

Validates that:
  1. The loss decreases over 600 epochs.
  2. Slot usage does not collapse (usage_entropy stays > 0.5 * log(16)).
  3. n_slots_used > 1 at convergence.
  4. Effective hop depth (weighted-mean hop index) is logged to check
     whether the model actually uses multi-hop propagation or stays at 0.
  5. ARI on the tuning slice 151673 is reported for rough orientation
     (NOT a held-out number -- 151673 is the single-slice-tuned slice
     for the baseline, so this is an upper-bound check only).

151673 is chosen because the existing baseline's per-seed ARI there is
~0.529 ± 0.054, giving a concrete reference point for sanity-checking
whether MSAP produces a useful embedding at all.

Saves to outputs/logs/msap_dlpfc_smoke.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.models.train_msap_model import train_msap_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "msap_dlpfc_smoke.json"
SMOKE_SLICE = "151673"
SMOKE_SEED = 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw = load_dlpfc_slice(SMOKE_SLICE)
    adata = preprocess_hvg(raw.copy())

    truth = adata.obs["ground_truth_layer"]
    mask_gt = truth.notna().to_numpy()
    n_layers = int(truth.nunique())
    coords = adata.obsm["spatial"]

    print(f"\n=== MSAP Smoke Test: slice {SMOKE_SLICE}, seed {SMOKE_SEED} ===")
    print(f"  n_spots={adata.n_obs}, n_layers={n_layers}")

    model, trained_adata, history = train_msap_model(
        adata.copy(),
        seed=SMOKE_SEED,
        device=device,
        epochs=600,
        log_every=100,
        verbose=True,
    )

    embedding = trained_adata.obsm["X_msap"]
    labels = cluster_embedding(embedding, n_layers, coords=coords, refine=True)
    ari = float(adjusted_rand_score(truth[mask_gt], np.asarray(labels)[mask_gt]))

    # Collapse check
    final = history[-1]
    max_entropy = final["max_entropy"]
    collapsed = (
        final["n_slots_used"] <= 1
        or final["usage_entropy"] < 0.5 * max_entropy
    )
    loss_decreased = history[-1]["recon_loss"] < history[0]["recon_loss"]

    print(f"\n=== Results ===")
    print(f"  ARI (slice {SMOKE_SLICE}, seed {SMOKE_SEED}): {ari:.4f}")
    print(f"  Baseline reference (per-seed, 5 seeds):    ~0.529 ± 0.054")
    print(f"  Loss decreased:   {loss_decreased}  "
          f"({history[0]['recon_loss']:.4f} -> {history[-1]['recon_loss']:.4f})")
    print(f"  Collapsed:        {collapsed}  "
          f"(n_slots_used={final['n_slots_used']}, "
          f"usage_entropy={final['usage_entropy']:.3f}/{max_entropy:.3f})")
    print(f"  Eff hop depth:    {final['effective_hop_depth']:.2f}  "
          f"(0=no propagation, {model.memory.max_hops}=max)")
    print(f"  Hop weight dist:  {[f'{w:.3f}' for w in final['hop_weights_mean']]}")

    result = {
        "slice": SMOKE_SLICE,
        "seed": SMOKE_SEED,
        "ari": ari,
        "baseline_reference_mean": 0.529,
        "baseline_reference_std": 0.054,
        "loss_decreased": loss_decreased,
        "collapsed": collapsed,
        "n_spots": int(adata.n_obs),
        "n_layers": n_layers,
        "history_summary": {
            "epoch_0": history[0],
            "epoch_100": next((h for h in history if h["epoch"] == 100), None),
            "epoch_300": next((h for h in history if h["epoch"] == 300), None),
            "epoch_final": history[-1],
        },
        "full_history": history,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
