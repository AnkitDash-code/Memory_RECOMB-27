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
from src.eval.clustering import cluster_embedding, consensus_cluster
from src.models.run_graphst import run_graphst
from src.models.train_count_model import train_count_model
from src.models.train_spatial_address import train_spatial_address_model

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "dlpfc_multislice_results.json"
KMEANS_INIT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "logs" / "dlpfc_multislice_results_kmeans_init.json"
)
TUNING_SLICE = "151673"


def _ari(truth, labels, mask):
    return float(adjusted_rand_score(truth[mask], np.asarray(labels)[mask]))


def evaluate_slice(sample, seeds, device, run_graphst_too=True, model="mse", kmeans_init=False):
    """model="mse" is the tuned, winning configuration (pure address propagation,
    memory_slots=32 -- see train_spatial_address_model's defaults). model="count"
    is the rejected NB/ZINB ablation, kept available for reproducing that negative
    result on demand, never as the default for a real evaluation. kmeans_init=True
    replaces the random memory_keys init with k-means centers of the initial
    per-spot queries -- an ablation, not evaluated at this scale before."""
    raw = load_dlpfc_slice(sample)
    adata = preprocess_hvg(raw.copy())

    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    if mask.sum() == 0:
        return None
    n_layers = int(truth.nunique())
    coords = adata.obsm["spatial"]

    results = {"sample": sample, "n_spots": int(adata.n_obs), "n_layers": n_layers}

    ours_labels = []
    ours_aris = []
    for seed in seeds:
        if model == "count":
            _, trained, _ = train_count_model(
                adata.copy(), seed=seed, device=device, verbose=False
            )
            embedding = trained.obsm["X_count_address"]
        else:
            _, trained, _ = train_spatial_address_model(
                adata.copy(), seed=seed, device=device, verbose=False,
                kmeans_init=kmeans_init,
            )
            embedding = trained.obsm["X_spatial_address"]
        labels = cluster_embedding(embedding, n_layers, coords=coords, refine=True)
        ours_labels.append(labels)
        ours_aris.append(_ari(truth, labels, mask))
    ours_consensus = consensus_cluster(ours_labels, n_layers)
    results["ours"] = {
        "per_seed": ours_aris,
        "mean": float(np.mean(ours_aris)),
        "std": float(np.std(ours_aris)),
        "consensus": _ari(truth, ours_consensus, mask),
    }

    if run_graphst_too:
        # Same seeds, same clustering protocol, and the SAME consensus
        # technique as "ours" -- if consensus-across-seeds helps, it must be
        # offered to the baseline too, or the comparison silently favors us.
        graphst_labels = []
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
            graphst_labels.append(labels)
            graphst_aris.append(_ari(gt, labels, gmask))
        graphst_consensus = consensus_cluster(graphst_labels, n_layers)
        results["graphst"] = {
            "per_seed": graphst_aris,
            "mean": float(np.mean(graphst_aris)),
            "std": float(np.std(graphst_aris)),
            "consensus": _ari(gt, graphst_consensus, gmask),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples", nargs="*", default=ALL_DLPFC_SAMPLES)
    parser.add_argument("--model", choices=["count", "mse"], default="mse")
    parser.add_argument("--skip-graphst", action="store_true")
    parser.add_argument("--kmeans-init", action="store_true",
                         help="Ablation: k-means codebook init instead of random. "
                              "Writes to a separate output file, not the main results.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(args.seeds))

    all_results = []
    for sample in args.samples:
        result = evaluate_slice(
            sample, seeds, device,
            run_graphst_too=not args.skip_graphst, model=args.model,
            kmeans_init=args.kmeans_init,
        )
        if result is None:
            continue
        all_results.append(result)
        line = f"{sample}: ours={result['ours']['mean']:.4f}+/-{result['ours']['std']:.4f}"
        line += f" (consensus={result['ours']['consensus']:.4f})"
        if "graphst" in result:
            line += f"  graphst={result['graphst']['mean']:.4f} (consensus={result['graphst']['consensus']:.4f})"
        print(line, flush=True)

    held_out = [r for r in all_results if r["sample"] != TUNING_SLICE]

    def summarize(rows, key, field="mean"):
        values = [r[key][field] for r in rows if key in r]
        return {
            "n_slices": len(values),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }

    summary = {
        "held_out_11_slices": {
            "ours": summarize(held_out, "ours"),
            "ours_consensus": summarize(held_out, "ours", field="consensus"),
            "graphst": summarize(held_out, "graphst"),
            "graphst_consensus": summarize(held_out, "graphst", field="consensus"),
        },
        "all_12_slices": {
            "ours": summarize(all_results, "ours"),
            "ours_consensus": summarize(all_results, "ours", field="consensus"),
            "graphst": summarize(all_results, "graphst"),
            "graphst_consensus": summarize(all_results, "graphst", field="consensus"),
        },
        "tuning_slice_excluded_from_headline": TUNING_SLICE,
    }

    print("\n=== HEADLINE (11 held-out slices; 151673 excluded as tuning slice) ===")
    for method in ("ours", "ours_consensus", "graphst", "graphst_consensus"):
        stats = summary["held_out_11_slices"][method]
        if stats["mean"] is not None:
            print(f"  {method:18s} ARI = {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n_slices']})")
    print("\n=== all 12 slices (for comparability with published tables) ===")
    for method in ("ours", "ours_consensus", "graphst", "graphst_consensus"):
        stats = summary["all_12_slices"][method]
        if stats["mean"] is not None:
            print(f"  {method:18s} ARI = {stats['mean']:.4f} +/- {stats['std']:.4f}  (n={stats['n_slices']})")

    output_path = KMEANS_INIT_OUTPUT_PATH if args.kmeans_init else OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"per_slice": all_results, "summary": summary}, indent=2))
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
