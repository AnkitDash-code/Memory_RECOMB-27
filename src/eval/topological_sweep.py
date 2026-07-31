"""som_sigma sweep (plan Section 3) + loss ablation (plan Section 4 item 4),
run together because the first gate attempt collapsed and they diagnose the
same question: is the collapse a tunable-hyperparameter problem or intrinsic
to the SOM loss as formulated?

Hypothesis under test, from reading the first collapsed run's instrumentation
(slots_used=1, key_cosine_similarity=0.995, expected_pos_std=0.0):

  The SOM topology loss has an UNOPPOSED degenerate minimum in this
  architecture. In SOM-VAE, reconstruction flows through the quantized
  codebook vector, so codebook entries must stay spread out to reconstruct
  well. Here, reconstruction flows through `memory_values`, a *separate*
  parameter -- `memory_keys` receive no spreading pressure from the
  reconstruction term at all, so the SOM term can freely collapse every key
  onto the query centroid.

  And the existing anti-collapse guard cannot catch this: when all keys are
  identical, the softmax over identical scores is UNIFORM, which is the
  MAXIMUM of usage_entropy. The guard is fully satisfied by the degenerate
  solution. usage_entropy prevents "every spot routes to one slot"; it does
  not prevent "every spot routes uniformly to all slots", which is equally
  uninformative and is what actually happened.

If the hypothesis is right, collapse should persist across som_sigma and only
disappear as lambda_som -> 0 (i.e. as the SOM mechanism is switched off),
and the ordinal-only configuration should behave like the Stage 13 baseline.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.eval.topological_axis_diagnostic import LAYER_ORDER, _layer_ordinal
from src.models.train_topological import train_topological_model

RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "topological_sweep.json"

# (label, lambda_som, lambda_ordinal, som_sigma)
CONFIGS = [
    ("ordinal-only",            0.0,    0.02,  1.5),
    ("som-only",                0.02,   0.0,   1.5),
    ("both (plan default)",     0.02,   0.02,  1.5),
    ("both, sigma=0.5",         0.02,   0.02,  0.5),
    ("both, sigma=1.0",         0.02,   0.02,  1.0),
    ("both, sigma=2.5",         0.02,   0.02,  2.5),
    ("both, lambda_som=0.002",  0.002,  0.02,  1.0),
    ("both, lambda_som=2e-4",   0.0002, 0.02,  1.0),
    ("neither (= Stage 13)",    0.0,    0.0,   1.5),
]


def evaluate(adata, truth, mask, depth, n_layers, coords, label,
             lambda_som, lambda_ordinal, som_sigma, seed, device):
    _, trained, history = train_topological_model(
        adata.copy(), seed=seed, device=device, verbose=False,
        lambda_som=lambda_som, lambda_ordinal=lambda_ordinal, som_sigma=som_sigma,
    )
    expected_pos = trained.obs["expected_slot_pos"].to_numpy()
    embedding = trained.obsm["X_topological_address"]

    labels = cluster_embedding(embedding, n_layers, coords=coords, refine=True)
    ari = float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))

    pos_std = float(np.std(expected_pos))
    if pos_std < 1e-9:
        rho = float("nan")  # constant axis -- correlation undefined, not zero
    else:
        rho = abs(float(spearmanr(expected_pos[mask], depth[mask]).statistic))

    return {
        "label": label,
        "lambda_som": lambda_som,
        "lambda_ordinal": lambda_ordinal,
        "som_sigma": som_sigma,
        "ari": ari,
        "abs_spearman_pos_vs_depth": rho,
        "expected_pos_std": pos_std,
        "n_slots_used": history[-1]["n_slots_used"],
        "key_cosine_similarity": history[-1]["key_cosine_similarity"],
        "final_recon_loss": history[-1]["recon_loss"],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = preprocess_hvg(load_dlpfc_slice("151673"))
    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    depth = _layer_ordinal(truth.to_numpy())
    n_layers = int(truth.nunique())
    coords = adata.obsm["spatial"]

    results = []
    header = (f"{'config':<26}{'ARI':>8}{'|rho|':>8}{'pos_std':>9}"
              f"{'slots':>7}{'key_cos':>9}{'recon':>8}")
    print(header)
    print("-" * len(header))
    for label, l_som, l_ord, sigma in CONFIGS:
        row = evaluate(adata, truth, mask, depth, n_layers, coords, label,
                       l_som, l_ord, sigma, seed=0, device=device)
        results.append(row)
        rho_str = "  n/a" if np.isnan(row["abs_spearman_pos_vs_depth"]) else f"{row['abs_spearman_pos_vs_depth']:.3f}"
        print(f"{label:<26}{row['ari']:>8.4f}{rho_str:>8}{row['expected_pos_std']:>9.4f}"
              f"{row['n_slots_used']:>7}{row['key_cosine_similarity']:>9.3f}"
              f"{row['final_recon_loss']:>8.4f}", flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
