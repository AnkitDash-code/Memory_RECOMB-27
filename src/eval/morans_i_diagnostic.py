"""Revised Section-2 diagnostic: spatial autocorrelation (Moran's I), not
spot-by-spot modality "disagreement".

A follow-up review of the original image_diagnostic.py result (subject-3
showed the *lowest* expr/image disagreement of the three subjects, the
opposite of what the dual-modality hypothesis needed) raised a sharper
reframing: the original diagnostic assumed histology would help by
*resolving conflicts* with expression, but the actual mechanism this
architecture would need is *signal rescue* -- histology staying spatially
coherent where transcriptomic signal has degraded (dropout/noise/low depth),
not disagreeing with it locally. A degraded expression vector doesn't
"conflict" with an intact image patch; it's just uninformative, which a
raw cosine-similarity-disagreement metric can't distinguish from genuine
conflict.

The sharper, falsifiable test: compute Global Moran's I (spatial
autocorrelation) separately for expression and for image features, per
slice. The hypothesis this plan needs to hold: on subject-3 slices,
expression Moran's I is LOW (spatially incoherent -- consistent with
degraded signal) while image Moran's I stays HIGH (tissue anatomy is
spatially intact regardless of transcriptomic quality). If subjects 1/2
don't show this same gap, that's the "signal rescue" pattern the
dual-modality architecture would need to exploit.

This is a pure data statistic -- no model training required, and reuses the
already-cached image features from image_diagnostic.py's run (no new frozen-
encoder forward pass needed unless force=True).
"""

import json
from pathlib import Path

import numpy as np

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import get_hvg_features, preprocess_hvg
from src.eval.analyze_multislice_variance import SUBJECTS
from src.models.image_encoder import encode_and_cache_features, get_frozen_encoder

RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "morans_i_diagnostic_results.json"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def morans_i(features, connectivities):
    """Global Moran's I per column of `features` (n_spots, n_features),
    using the raw (non-row-normalized, no self-loops) spatial adjacency --
    the standard convention for Moran's I, distinct from the row-stochastic
    D^-1(A+I) matrix used elsewhere in this project for address propagation.

    I_j = (n / S0) * (x_j^T W x_j) / (x_j^T x_j),  x_j mean-centered.

    Vectorized across all columns at once via one sparse-dense matmul.
    """
    import scipy.sparse as sp

    W = sp.csr_matrix(connectivities)
    n = W.shape[0]
    s0 = W.sum()

    x = np.asarray(features, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)

    wxc = W @ xc  # (n, d)
    numerator = (xc * wxc).sum(axis=0)  # (d,)
    denominator = (xc**2).sum(axis=0)  # (d,)
    denominator[denominator == 0] = np.nan

    return (n / s0) * numerator / denominator


def run_slice(sample, device, encoder, backbone="resnet18"):
    raw = load_dlpfc_slice(sample)
    adata = preprocess_hvg(raw.copy())
    conn = adata.obsp["spatial_connectivities"]

    expr_features = get_hvg_features(adata)
    patches_path = Path(__file__).resolve().parents[2] / "outputs" / "cache" / f"patches_{sample}.npy"
    patches = np.load(patches_path)
    img_features = encode_and_cache_features(
        patches, sample, encoder=encoder, device=device, backbone=backbone
    )

    expr_moran = morans_i(expr_features, conn)
    img_moran = morans_i(img_features, conn)

    return {
        "sample": sample,
        "n_spots": int(adata.n_obs),
        "mean_expr_moran": float(np.nanmean(expr_moran)),
        "median_expr_moran": float(np.nanmedian(expr_moran)),
        "mean_img_moran": float(np.nanmean(img_moran)),
        "median_img_moran": float(np.nanmedian(img_moran)),
    }


def main(backbone="resnet18"):
    import torch

    from src.models.image_encoder import get_dinov2_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_dinov2_encoder(device=device) if backbone == "dinov2" else get_frozen_encoder(device=device)

    results = []
    for subject, samples in SUBJECTS.items():
        for sample in samples:
            row = run_slice(sample, device, encoder, backbone=backbone)
            row["subject"] = subject
            results.append(row)
            print(f"{sample} ({subject}): expr_moran(mean/median)={row['mean_expr_moran']:.4f}/"
                  f"{row['median_expr_moran']:.4f}  img_moran(mean/median)={row['mean_img_moran']:.4f}/"
                  f"{row['median_img_moran']:.4f}", flush=True)

    print("\n=== per-subject summary ===")
    print(f"{'subject':<10}{'mean_expr_moran':>18}{'mean_img_moran':>18}{'gap (img-expr)':>16}")
    subject_summary = {}
    for subject in SUBJECTS:
        rows = [r for r in results if r["subject"] == subject]
        mean_expr = float(np.mean([r["mean_expr_moran"] for r in rows]))
        mean_img = float(np.mean([r["mean_img_moran"] for r in rows]))
        subject_summary[subject] = {
            "mean_expr_moran": mean_expr, "mean_img_moran": mean_img, "gap": mean_img - mean_expr,
        }
        print(f"{subject:<10}{mean_expr:>18.4f}{mean_img:>18.4f}{mean_img - mean_expr:>16.4f}")

    suffix = "" if backbone == "resnet18" else f"_{backbone}"
    results_path = RESULTS_PATH.parent / f"morans_i_diagnostic_results{suffix}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"per_slice": results, "per_subject": subject_summary}, indent=2))
    print(f"\nSaved {results_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot(results, subject_summary, suffix)


def _plot(results, subject_summary, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"subject1": "#2c5f8a", "subject2": "#4f8fc0", "subject3": "#c0392b"}
    for subject in SUBJECTS:
        rows = [r for r in results if r["subject"] == subject]
        xs = [r["mean_expr_moran"] for r in rows]
        ys = [r["mean_img_moran"] for r in rows]
        ax.scatter(xs, ys, label=subject, color=colors[subject], s=80, alpha=0.8)
        for r in rows:
            ax.annotate(r["sample"], (r["mean_expr_moran"], r["mean_img_moran"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, label="img Moran's I = expr Moran's I")
    ax.set_xlabel("Mean expression Moran's I (spatial autocorrelation)")
    ax.set_ylabel("Mean image-feature Moran's I (spatial autocorrelation)")
    backbone_label = "ResNet18" if not suffix else "DINOv2"
    ax.set_title(f"Signal-rescue hypothesis ({backbone_label}): points above the line = image more\n"
                 "spatially coherent than expression")
    ax.legend()
    fig.tight_layout()
    fig_name = f"morans_i_diagnostic_scatter{suffix}.png"
    fig.savefig(FIGURES_DIR / fig_name, dpi=150)
    plt.close(fig)
    print(f"Saved outputs/figures/{fig_name}")


if __name__ == "__main__":
    import sys

    main(backbone=sys.argv[1] if len(sys.argv) > 1 else "resnet18")
