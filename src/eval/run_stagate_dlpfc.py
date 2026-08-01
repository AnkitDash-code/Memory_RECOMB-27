"""STAGATE across all 12 DLPFC slices, same protocol as every other method here.

Phase B3. STAGATE was recorded in this project as "blocked on Windows"
(torch_sparse having no wheel for torch 2.11.0+cu128). That claim was STALE:
PyG's wheel index now publishes builds through torch 2.13.0, including
torch-sparse 0.6.18+pt211cu128, so STAGATE runs locally after all and does not
need Colab. Re-checking an inherited blocker before routing around it turned
out to be worth more than the workaround.

(Garfield remains genuinely blocked here: it depends on pybedtools -> pysam,
which needs htslib and has no Windows build. That one is a real platform
limit, not a stale wheel index, and still needs Colab or WSL.)

Same evaluation contract as run_dlpfc_multislice.py so the numbers drop
straight into the results table: 5 seeds, the shared mclust-equivalent +
spatial-refinement clustering, consensus across seeds, and 151673 excluded
from the headline mean as the tuning slice.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import ALL_DLPFC_SAMPLES, load_dlpfc_slice
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.run_stagate import run_stagate

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "stagate_dlpfc_results.json"
TUNING_SLICE = "151673"
SEEDS = [0, 1, 2, 3, 4]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for sample in ALL_DLPFC_SAMPLES:
        raw = load_dlpfc_slice(sample)
        truth = raw.obs["ground_truth_layer"]
        mask = truth.notna().to_numpy()
        n_layers = int(truth.nunique())
        truth_np = truth.to_numpy()

        labels_per_seed, aris = [], []
        for seed in SEEDS:
            out = run_stagate(raw.copy(), device=device, random_seed=seed)
            labels = cluster_embedding(out.obsm["STAGATE"], n_layers,
                                       coords=out.obsm["spatial"], refine=True)
            labels_per_seed.append(labels)
            aris.append(float(adjusted_rand_score(truth_np[mask], np.asarray(labels)[mask])))

        cons = consensus_cluster(labels_per_seed, n_layers)
        row = {
            "sample": sample,
            "per_seed": aris,
            "mean": float(np.mean(aris)),
            "std": float(np.std(aris)),
            "consensus": float(adjusted_rand_score(truth_np[mask], np.asarray(cons)[mask])),
        }
        results.append(row)
        print(f"{sample}: stagate={row['mean']:.4f}+/-{row['std']:.4f} "
              f"(consensus={row['consensus']:.4f})", flush=True)

    held = [r for r in results if r["sample"] != TUNING_SLICE]
    summary = {
        "held_out_11_slices": {
            "per_seed_mean": float(np.mean([r["mean"] for r in held])),
            "per_seed_std": float(np.std([r["mean"] for r in held])),
            "consensus_mean": float(np.mean([r["consensus"] for r in held])),
            "consensus_std": float(np.std([r["consensus"] for r in held])),
        },
        "all_12_slices": {
            "per_seed_mean": float(np.mean([r["mean"] for r in results])),
            "consensus_mean": float(np.mean([r["consensus"] for r in results])),
        },
    }
    h = summary["held_out_11_slices"]
    print(f"\n=== STAGATE held-out (11 slices) ===")
    print(f"  per-seed  ARI = {h['per_seed_mean']:.4f} +/- {h['per_seed_std']:.4f}")
    print(f"  consensus ARI = {h['consensus_mean']:.4f} +/- {h['consensus_std']:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"per_slice": results, "summary": summary}, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
