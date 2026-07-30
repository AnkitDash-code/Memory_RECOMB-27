"""Final evaluation: all 12 DLPFC slices, multi-seed, identical clustering protocol.

Methodology notes that matter for the credibility of these numbers:

* **Tuning/reporting split.** Slice 151673 was used to choose hyperparameters, so
  reporting a mean that includes it would leak. The headline figure is the mean
  over the other 11 slices; the all-12 mean is also printed for comparability
  with published tables, and the two are reported separately, never merged.
* **Identical protocol for every method.** All embeddings go through the same
  mclust-equivalent + spatial-refinement clustering with K set to that slice's
  true number of annotated layers. Providing K is standard in this benchmark.
* **All seeds reported.** Mean and std over seeds, never best-of-N.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import ALL_DLPFC_SAMPLES, load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import cluster_embedding
from src.models.run_graphst import run_graphst
from src.models.train_count_model import train_count_model
from src.models.train_spatial_address import train_spatial_address_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "dlpfc_multislice_results.json"
TUNING_SLICE = "151673"


def _ari(truth, labels, mask):
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def evaluate_slice(sample, seeds, device, run_graphst_too=True, model="mse"):
    """model="mse" is the tuned, winning configuration (pure address propagation,
    memory_slots=32 -- see train_spatial_address_model's defaults). model="count"
    is the rejected NB/ZINB ablation, kept available for reproducing that negative
    result on demand, never as the default for a real evaluation."""
    raw = load_dlpfc_slice(sample)
    adata = preprocess_hvg(raw.copy())

    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    if mask.sum() == 0:
        return None
    n_layers = int(truth.nunique())
    coords = adata.obsm["spatial"]

    results = {"sample": sample, "n_spots": int(adata.n_obs), "n_layers": n_layers}

    ours = []
    for seed in seeds:
        if model == "count":
            _, trained, _ = train_count_model(
                adata.copy(), seed=seed, device=device, verbose=False
            )
            embedding = trained.obsm["X_count_address"]
        else:
            _, trained, _ = train_spatial_address_model(
                adata.copy(), seed=seed, device=device, verbose=False,
            )
            embedding = trained.obsm["X_spatial_address"]
        labels = cluster_embedding(embedding, n_layers, coords=coords, refine=True)
        ours.append(_ari(truth, labels, mask))
    results["ours"] = {"per_seed": ours, "mean": float(np.mean(ours)), "std": float(np.std(ours))}

    if run_graphst_too:
        # Same seeds, same clustering protocol as "ours" -- comparing a 5-seed
        # mean against GraphST's single default seed was measured on 151673 to
        # be unfair (its default seed wasn't even its best of 5).
        graphst_aris = []
        for seed in seeds:
            graphst_adata = run_graphst(
                raw.copy(), n_clusters=n_layers, device=device,
                random_seed=seed, cluster=False,
            )
            labels = cluster_embedding(
                graphst_adata.obsm["emb"], n_layers,
                coords=graphst_adata.obsm["spatial"], refine=True,
            )
            gt = truth.reindex(graphst_adata.obs_names)
            gmask = gt.notna().to_numpy()
            graphst_aris.append(_ari(gt, labels, gmask))
        results["graphst"] = {
            "per_seed": graphst_aris,
            "mean": float(np.mean(graphst_aris)),
            "std": float(np.std(graphst_aris)),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples", nargs="*", default=ALL_DLPFC_SAMPLES)
    parser.add_argument("--model", choices=["count", "mse"], default="mse")
    parser.add_argument("--skip-graphst", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seeds))

    all_results = []
    for sample in args.samples:
        result = evaluate_slice(
            sample, seeds, device,
            run_graphst_too=not args.skip_graphst, model=args.model,
        )
        if result is None:
            continue
        all_results.append(result)
        line = f"{sample}: ours={result['ours']['mean']:.4f}+/-{result['ours']['std']:.4f}"
        if "graphst" in result:
            line += f"  graphst={result['graphst']['mean']:.4f}"
        print(line, flush=True)

    held_out = [r for r in all_results if r["sample"] != TUNING_SLICE]

    def summarize(rows, key):
        values = [r[key]["mean"] for r in rows if key in r]
        return {
            "n_slices": len(values),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }

    summary = {
        "held_out_11_slices": {
            "ours": summarize(held_out, "ours"),
            "graphst": summarize(held_out, "graphst"),
        },
        "all_12_slices": {
            "ours": summarize(all_results, "ours"),
            "graphst": summarize(all_results, "graphst"),
        },
        "tuning_slice_excluded_from_headline": TUNING_SLICE,
    }

    print("\n=== HEADLINE (11 held-out slices; 151673 excluded as tuning slice) ===")
    for method in ("ours", "graphst"):
        stats = summary["held_out_11_slices"][method]
        if stats["mean"] is not None:
            print(f"  {method:8s} ARI = {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n_slices']})")
    print("\n=== all 12 slices (for comparability with published tables) ===")
    for method in ("ours", "graphst"):
        stats = summary["all_12_slices"][method]
        if stats["mean"] is not None:
            print(f"  {method:8s} ARI = {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n_slices']})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"per_slice": all_results, "summary": summary}, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
