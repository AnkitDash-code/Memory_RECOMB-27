"""GATE for the Topologically-Ordered Memory (TOM) plan, Section 4 step 1.

The cheap falsification test, run before any multi-seed or full-protocol
evaluation: does the model's emergent 1D ordinal axis (`expected_slot_pos`)
actually track true cortical layer order? If not, the topology isn't tracking
real structure and nothing downstream in the plan will help -- stop, same
stopping discipline that correctly halted the multimodal plan at Stages 16/17.

DEVIATION FROM THE PLAN, deliberate: the plan specifies the crop dataset for
this step, but crop (squidpy mouse Visium H&E) has NO ground-truth layer
annotation, so the correlation being tested is uncomputable there. Run on
DLPFC 151673 instead -- the designated tuning slice, so using it for a
development check leaks nothing into the held-out result.

TWO CONTROLS the plan does not specify, without which a positive result would
be uninterpretable:

  (a) PC1 of TOM's own embedding vs. layer order. Cortical depth is plausibly
      the dominant axis of variation in ANY reasonable embedding of this
      tissue, so a "high" correlation for expected_pos means little unless it
      is compared against the axis you would get for free from PCA.
  (b) PC1 of the existing Stage 13 (non-topological) model's embedding vs.
      layer order. This is the real bar: if the current model's free PCA axis
      already tracks layer order just as well, the explicit ordinal machinery
      is adding nothing and the plan's premise is wrong even if (a) looks good.

Spearman (rank) correlation throughout, since only the ORDER is claimed, not
linearity of spacing. Absolute value is reported because the learned axis
direction is arbitrary -- the model has no reason to prefer L1->WM over
WM->L1, and either is an equally successful outcome.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.models.train_spatial_address import train_spatial_address_model
from src.models.train_topological import train_topological_model

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "logs" / "topological_axis_diagnostic.json"
)
FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"

LAYER_ORDER = {
    "Layer1": 1, "Layer2": 2, "Layer3": 3, "Layer4": 4,
    "Layer5": 5, "Layer6": 6, "WM": 7,
}


def _layer_ordinal(truth):
    return np.array([LAYER_ORDER.get(v, np.nan) for v in truth], dtype=float)


def _abs_spearman(a, b):
    rho, p = spearmanr(a, b)
    return abs(float(rho)), float(p)


def run(sample="151673", seed=0, device=None, lambda_som=0.02, lambda_ordinal=0.02,
        som_sigma=1.5, verbose=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = preprocess_hvg(load_dlpfc_slice(sample))

    truth = adata.obs["ground_truth_layer"]
    mask = truth.notna().to_numpy()
    depth = _layer_ordinal(truth.to_numpy())

    _, tom_adata, history = train_topological_model(
        adata.copy(), seed=seed, device=device, verbose=verbose,
        lambda_som=lambda_som, lambda_ordinal=lambda_ordinal, som_sigma=som_sigma,
    )
    expected_pos = tom_adata.obs["expected_slot_pos"].to_numpy()
    tom_embedding = tom_adata.obsm["X_topological_address"]

    _, base_adata, _ = train_spatial_address_model(
        adata.copy(), seed=seed, device=device, verbose=False
    )
    base_embedding = base_adata.obsm["X_spatial_address"]

    tom_pc1 = PCA(n_components=1).fit_transform(tom_embedding)[:, 0]
    base_pc1 = PCA(n_components=1).fit_transform(base_embedding)[:, 0]

    rho_pos, p_pos = _abs_spearman(expected_pos[mask], depth[mask])
    rho_tom_pc1, p_tom_pc1 = _abs_spearman(tom_pc1[mask], depth[mask])
    rho_base_pc1, p_base_pc1 = _abs_spearman(base_pc1[mask], depth[mask])

    result = {
        "sample": sample,
        "seed": seed,
        "lambda_som": lambda_som,
        "lambda_ordinal": lambda_ordinal,
        "som_sigma": som_sigma,
        "n_annotated_spots": int(mask.sum()),
        "spearman_expected_pos_vs_depth": rho_pos,
        "p_expected_pos": p_pos,
        "spearman_tom_pc1_vs_depth": rho_tom_pc1,
        "p_tom_pc1": p_tom_pc1,
        "spearman_baseline_pc1_vs_depth": rho_base_pc1,
        "p_baseline_pc1": p_base_pc1,
        "final_n_slots_used": history[-1]["n_slots_used"],
        "final_expected_pos_std": history[-1]["expected_pos_std"],
        "final_key_cosine_similarity": history[-1]["key_cosine_similarity"],
    }
    return result, expected_pos, depth, mask, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result, expected_pos, depth, mask, history = run(device=device)

    print("\n=== TOM ordinal-axis gate (DLPFC 151673, seed 0) ===")
    print(f"  |Spearman| expected_pos      vs true layer order : "
          f"{result['spearman_expected_pos_vs_depth']:.4f}  (p={result['p_expected_pos']:.2e})")
    print(f"  |Spearman| TOM embedding PC1  vs true layer order : "
          f"{result['spearman_tom_pc1_vs_depth']:.4f}  (control a)")
    print(f"  |Spearman| Stage-13 base PC1  vs true layer order : "
          f"{result['spearman_baseline_pc1_vs_depth']:.4f}  (control b)")
    print(f"  slots used at end: {result['final_n_slots_used']}  "
          f"expected_pos std: {result['final_expected_pos_std']:.4f}  "
          f"key_cos_sim: {result['final_key_cosine_similarity']:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"result": result, "history": history}, indent=2))
    print(f"\nSaved {RESULTS_PATH}")

    _plot(expected_pos, depth, mask, result)


def _plot(expected_pos, depth, mask, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = sorted(set(depth[mask]))
    data = [expected_pos[mask][depth[mask] == d] for d in layers]
    names = [k for k, v in sorted(LAYER_ORDER.items(), key=lambda kv: kv[1]) if v in layers]

    ax.boxplot(data, tick_labels=names, showfliers=False)
    ax.set_xlabel("True cortical layer (ordered L1 -> WM)")
    ax.set_ylabel("Learned expected slot position (normalized)")
    ax.set_title(
        "TOM gate: does the learned 1D memory axis track cortical depth?\n"
        f"|Spearman| = {result['spearman_expected_pos_vs_depth']:.3f} "
        f"(baseline PC1 control: {result['spearman_baseline_pc1_vs_depth']:.3f})"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "topological_axis_gate.png", dpi=150)
    plt.close(fig)
    print("Saved outputs/figures/topological_axis_gate.png")


if __name__ == "__main__":
    main()
