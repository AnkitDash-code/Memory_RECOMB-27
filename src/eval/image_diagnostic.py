"""Section 2 falsification test for the dual-modality (expression + morphology)
memory-addressing plan -- run BEFORE writing the dual-memory architecture.

Hypothesis under test: current expression-only model errors concentrate where
expression similarity and morphology (H&E) similarity *disagree* among spatial
neighbors, and this effect should be more pronounced on the subject-3 DLPFC
slices (151673-676), the persistent weak point of every fix tried so far
(Stages 8, 9, 11, 13, 14, 15) even though none of those fixes ever touched the
image modality.

Per spot i, restricted to its spatial neighbors N(i) (from
adata.obsp['spatial_connectivities'], the same graph the model already
propagates over):

  neighbor_expr_sim(i) = mean_j cosine_sim(expr_i, expr_j),  j in N(i)
  neighbor_img_sim(i)  = mean_j cosine_sim(img_i, img_j),    j in N(i)

Both z-scored within-slice (raw cosine similarity ranges are not directly
comparable across modalities with different feature dimensionalities and
distributions) before taking:

  disagreement(i) = |z(neighbor_img_sim(i)) - z(neighbor_expr_sim(i))|

model_error(i): the current best model (single seed, current defaults) is
trained and clustered exactly as in the real evaluation harness; predicted
cluster IDs are majority-vote-mapped to ground-truth layers (standard
cluster-purity mapping), and error(i) = 1 if the mapped prediction disagrees
with the true label, else 0.

Gate: if subject 3 does not show elevated disagreement AND elevated error
relative to subjects 1/2, the dual-memory architecture should not be built --
see outputs/logs/stage2_progress.md for the logged conclusion.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score

from src.data.extract_patches import extract_and_cache_patches
from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import get_hvg_features, preprocess_hvg
from src.eval.analyze_multislice_variance import SUBJECTS
from src.eval.clustering import cluster_embedding
from src.models.image_encoder import encode_and_cache_features, get_frozen_encoder
from src.models.train_spatial_address import train_spatial_address_model

RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "image_diagnostic_results.json"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def _neighbor_mean_similarity(sim_matrix, connectivities):
    """Per-spot mean similarity to its spatial neighbors (excludes self)."""
    conn = connectivities.tocsr()
    n = conn.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neighbors = conn.indices[conn.indptr[i]:conn.indptr[i + 1]]
        if len(neighbors) == 0:
            out[i] = np.nan
            continue
        out[i] = sim_matrix[i, neighbors].mean()
    return out


def _zscore(x):
    valid = ~np.isnan(x)
    mean, std = x[valid].mean(), x[valid].std()
    std = std if std > 0 else 1.0
    return (x - mean) / std


def _cluster_purity_error(predicted_labels, true_labels):
    """Map each predicted cluster to its majority true label, then return a
    per-spot binary error indicator (1 = mismatch after mapping)."""
    predicted_labels = np.asarray(predicted_labels)
    true_labels = np.asarray(true_labels, dtype=object)
    mapping = {}
    for cluster in np.unique(predicted_labels):
        mask = predicted_labels == cluster
        values, counts = np.unique(true_labels[mask], return_counts=True)
        mapping[cluster] = values[np.argmax(counts)]
    mapped = np.array([mapping[c] for c in predicted_labels], dtype=object)
    return (mapped != true_labels).astype(int)


def run_slice(sample, device, encoder):
    raw = load_dlpfc_slice(sample)
    adata = preprocess_hvg(raw.copy())

    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    n_layers = int(truth.nunique())
    coords = adata.obsm["spatial"]

    expr_features = get_hvg_features(adata)
    patches = extract_and_cache_patches(raw, sample)
    img_features = encode_and_cache_features(patches, sample, encoder=encoder, device=device)

    expr_sim = cosine_similarity(expr_features)
    img_sim = cosine_similarity(img_features)

    conn = adata.obsp["spatial_connectivities"]
    neighbor_expr = _neighbor_mean_similarity(expr_sim, conn)
    neighbor_img = _neighbor_mean_similarity(img_sim, conn)
    disagreement = np.abs(_zscore(neighbor_img) - _zscore(neighbor_expr))

    _, trained, _ = train_spatial_address_model(adata.copy(), seed=0, device=device, verbose=False)
    labels = np.asarray(cluster_embedding(trained.obsm["X_spatial_address"], n_layers, coords=coords, refine=True))
    ari = float(adjusted_rand_score(truth[mask], labels[mask]))

    # Ground truth is NaN for a handful of unannotated spots (see spatialLIBD
    # source) -- restrict the purity mapping and error/disagreement comparison
    # to annotated spots only, same as every other metric in this project.
    # A spot can also have no spatial neighbors (isolated node in the kNN
    # graph) -> disagreement is NaN there; drop those too before correlating.
    error = _cluster_purity_error(labels[mask], truth.to_numpy()[mask])
    disagreement_masked = disagreement[mask]
    finite = ~np.isnan(disagreement_masked)

    return {
        "sample": sample,
        "n_spots": int(adata.n_obs),
        "mean_disagreement": float(np.nanmean(disagreement_masked)),
        "mean_error": float(error.mean()),
        "ari": ari,
        "per_spot_correlation": float(np.corrcoef(disagreement_masked[finite], error[finite])[0, 1]),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_frozen_encoder(device=device)

    sample_to_subject = {s: subj for subj, samples in SUBJECTS.items() for s in samples}
    results = []
    for subject, samples in SUBJECTS.items():
        for sample in samples:
            row = run_slice(sample, device, encoder)
            row["subject"] = subject
            results.append(row)
            print(f"{sample} ({subject}): mean_disagreement={row['mean_disagreement']:.4f}  "
                  f"mean_error={row['mean_error']:.4f}  ari={row['ari']:.4f}  "
                  f"per_spot_corr={row['per_spot_correlation']:.4f}", flush=True)

    print("\n=== per-subject summary ===")
    print(f"{'subject':<10}{'mean_disagreement':>20}{'mean_error':>14}{'mean_ari':>10}")
    subject_summary = {}
    for subject in SUBJECTS:
        rows = [r for r in results if r["subject"] == subject]
        mean_dis = float(np.mean([r["mean_disagreement"] for r in rows]))
        mean_err = float(np.mean([r["mean_error"] for r in rows]))
        mean_ari = float(np.mean([r["ari"] for r in rows]))
        subject_summary[subject] = {
            "mean_disagreement": mean_dis, "mean_error": mean_err, "mean_ari": mean_ari,
        }
        print(f"{subject:<10}{mean_dis:>20.4f}{mean_err:>14.4f}{mean_ari:>10.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"per_slice": results, "per_subject": subject_summary}, indent=2))
    print(f"\nSaved {RESULTS_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot(results, subject_summary)


def _plot(results, subject_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"subject1": "#2c5f8a", "subject2": "#4f8fc0", "subject3": "#c0392b"}
    for subject, rows in [(s, [r for r in results if r["subject"] == s]) for s in SUBJECTS]:
        xs = [r["mean_disagreement"] for r in rows]
        ys = [r["mean_error"] for r in rows]
        ax.scatter(xs, ys, label=subject, color=colors[subject], s=80, alpha=0.8)
        for r in rows:
            ax.annotate(r["sample"], (r["mean_disagreement"], r["mean_error"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        sx, sy = subject_summary[subject]["mean_disagreement"], subject_summary[subject]["mean_error"]
        ax.scatter([sx], [sy], color=colors[subject], s=300, marker="X", edgecolor="black", linewidth=1.5)

    ax.set_xlabel("Mean expr/image neighbor-similarity disagreement (z-scored)")
    ax.set_ylabel("Mean model error rate (cluster-purity mismatch)")
    ax.set_title("Section 2 diagnostic: does modality disagreement predict model error?\n(X = per-subject mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "image_diagnostic_scatter.png", dpi=150)
    plt.close(fig)
    print("Saved outputs/figures/image_diagnostic_scatter.png")


if __name__ == "__main__":
    main()
