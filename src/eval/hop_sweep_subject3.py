"""Phase A: leakage-safe per-subject `n_hops` sweep on subject 3, with a
boundary-vs-interior breakdown.

Hypothesis (from Stage 19): subject 3 has by far the richest per-spot signal
(2058 genes/spot vs 1324 for subject 1) yet the worst ARI, and GraphST handles
it fine. One concrete suspect is that `n_hops=4` address propagation
over-smooths genuinely separable layers precisely where signal is strongest.
`n_hops` was cross-validated globally (Stage 11), never per-subject.

THE ACTUAL TEST IS THE BREAKDOWN, NOT THE HEADLINE ARI. If over-smoothing is
the mechanism, lowering the hop count should specifically improve
BOUNDARY-ADJACENT spots (within 2 graph hops of a differently-labelled spot)
while leaving INTERIOR spots flat or slightly worse. An aggregate ARI bump
alone cannot distinguish that mechanism from noise -- so the stopping rule is
written against the breakdown.

LEAKAGE-SAFE SPLIT:
  * selection slice: 151673 only. It is already the project's global tuning
    slice and therefore already "burned" -- reusing it costs nothing, whereas
    selecting on any other subject-3 slice would contaminate a currently-clean
    held-out slice.
  * held-out evaluation: 151674, 151675, 151676, none of which are used to
    choose the hop count.
  The full sweep is still RUN on all four slices, because the per-slice curve
  is the diagnostic; only the *selection* is restricted to 151673.

Honest caveat, stated because this project has been burned by it before
(Stage 8): selecting on a single slice is exactly the failure mode that made
`memory_slots=32` look good. The mitigation here is that the selection slice
is pre-burned and the reported number comes from three genuinely unseen
slices -- but a single-slice selection is still weaker evidence than the
3-slice cross-validation used for `memory_slots`/`lambda_usage`, and the
decision should be weighted accordingly.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.boundary_mask import boundary_mask
from src.eval.clustering import cluster_embedding
from src.models.train_spatial_address import train_spatial_address_model

RESULTS_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "hop_sweep_subject3.json"

SELECTION_SLICE = "151673"          # already burned as the global tuning slice
HELD_OUT_SLICES = ["151674", "151675", "151676"]
ALL_SUBJECT3 = [SELECTION_SLICE] + HELD_OUT_SLICES

HOP_CANDIDATES = [1, 2, 3, 4, 5, 6]
SEEDS = [0, 1, 2, 3, 4]
GLOBAL_DEFAULT_HOPS = 4
BOUNDARY_RADIUS = 2


def evaluate_slice(sample, device):
    adata = preprocess_hvg(load_dlpfc_slice(sample))
    truth = adata.obs["ground_truth_layer"]
    annotated = truth.notna().to_numpy()
    n_layers = int(truth.nunique())
    coords = adata.obsm["spatial"]

    is_boundary = boundary_mask(
        truth.to_numpy(), adata.obsp["spatial_connectivities"],
        radius=BOUNDARY_RADIUS, valid=annotated,
    )
    boundary_sel = annotated & is_boundary
    interior_sel = annotated & ~is_boundary

    truth_np = truth.to_numpy()
    rows = []
    for n_hops in HOP_CANDIDATES:
        overall, bnd, inter = [], [], []
        for seed in SEEDS:
            _, trained, _ = train_spatial_address_model(
                adata.copy(), n_hops=n_hops, seed=seed, device=device, verbose=False
            )
            labels = np.asarray(cluster_embedding(
                trained.obsm["X_spatial_address"], n_layers, coords=coords, refine=True
            ))
            overall.append(adjusted_rand_score(truth_np[annotated], labels[annotated]))
            bnd.append(adjusted_rand_score(truth_np[boundary_sel], labels[boundary_sel]))
            inter.append(adjusted_rand_score(truth_np[interior_sel], labels[interior_sel]))

        rows.append({
            "sample": sample,
            "n_hops": n_hops,
            "overall_mean": float(np.mean(overall)), "overall_std": float(np.std(overall)),
            "boundary_mean": float(np.mean(bnd)), "boundary_std": float(np.std(bnd)),
            "interior_mean": float(np.mean(inter)), "interior_std": float(np.std(inter)),
            "n_boundary_spots": int(boundary_sel.sum()),
            "n_interior_spots": int(interior_sel.sum()),
        })
        r = rows[-1]
        print(f"  {sample} n_hops={n_hops}: overall={r['overall_mean']:.4f}+/-{r['overall_std']:.4f}"
              f"  boundary={r['boundary_mean']:.4f}  interior={r['interior_mean']:.4f}", flush=True)
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = []
    for sample in ALL_SUBJECT3:
        print(f"\n=== {sample} ({'SELECTION' if sample == SELECTION_SLICE else 'held-out'}) ===",
              flush=True)
        all_rows.extend(evaluate_slice(sample, device))

    def get(sample, n_hops, key):
        return next(r[key] for r in all_rows if r["sample"] == sample and r["n_hops"] == n_hops)

    # --- selection, on the pre-burned slice only ---------------------------
    sel_scores = {h: get(SELECTION_SLICE, h, "overall_mean") for h in HOP_CANDIDATES}
    selected = max(sel_scores, key=sel_scores.get)
    print(f"\n=== selection on {SELECTION_SLICE} ===")
    for h in HOP_CANDIDATES:
        print(f"  n_hops={h}: {sel_scores[h]:.4f}{'   <-- selected' if h == selected else ''}")

    # --- held-out comparison: selected vs global default -------------------
    print(f"\n=== held-out ({', '.join(HELD_OUT_SLICES)}) ===")
    print(f"{'metric':<12}{'global n_hops=4':>18}{f'selected n_hops={selected}':>22}{'delta':>10}")
    summary = {}
    for key, label in [("overall_mean", "overall"), ("boundary_mean", "boundary"),
                       ("interior_mean", "interior")]:
        base = float(np.mean([get(s, GLOBAL_DEFAULT_HOPS, key) for s in HELD_OUT_SLICES]))
        new = float(np.mean([get(s, selected, key) for s in HELD_OUT_SLICES]))
        summary[label] = {"global_n_hops_4": base, f"selected_n_hops_{selected}": new,
                          "delta": new - base}
        print(f"{label:<12}{base:>18.4f}{new:>22.4f}{new - base:>10.4f}")

    # --- the mechanistic test ---------------------------------------------
    #
    # A sign check alone is NOT sufficient and an earlier version of this
    # function got it wrong by using one: with per-slice seed std around
    # 0.03-0.06, a mean delta of +0.005 has the "right" sign while being pure
    # noise. The verdict therefore requires BOTH that the effect exceeds seed
    # noise AND that it is consistent in direction across the held-out slices
    # -- the plan's stopping rule explicitly says not to adopt a hop count
    # that only moves aggregate ARI by noise.
    b_delta = summary["boundary"]["delta"]
    i_delta = summary["interior"]["delta"]

    per_slice_boundary = [get(s, selected, "boundary_mean") - get(s, GLOBAL_DEFAULT_HOPS, "boundary_mean")
                          for s in HELD_OUT_SLICES]
    typical_noise = float(np.mean([
        max(get(s, selected, "boundary_std"), get(s, GLOBAL_DEFAULT_HOPS, "boundary_std"))
        for s in HELD_OUT_SLICES
    ]))
    exceeds_noise = abs(b_delta) > typical_noise
    consistent = all(d > 0 for d in per_slice_boundary) or all(d < 0 for d in per_slice_boundary)
    per_slice_best = {s: max(HOP_CANDIDATES, key=lambda h: get(s, h, "overall_mean"))
                      for s in HELD_OUT_SLICES}
    default_already_best = sum(1 for b in per_slice_best.values() if b == GLOBAL_DEFAULT_HOPS)

    print(f"\nMECHANISTIC TEST: boundary delta ({b_delta:+.4f}) vs interior delta ({i_delta:+.4f})")
    print(f"  per-slice boundary deltas: {[round(d, 4) for d in per_slice_boundary]}")
    print(f"  typical per-slice seed noise (boundary): {typical_noise:.4f}")
    print(f"  exceeds noise: {exceeds_noise}   direction-consistent across slices: {consistent}")
    print(f"  per-slice best hop: {per_slice_best}  "
          f"(global default already best on {default_already_best}/{len(HELD_OUT_SLICES)})")

    if selected == GLOBAL_DEFAULT_HOPS:
        verdict = ("NOT SUPPORTED: selection reproduced the global default, so there is no "
                   "per-subject hop effect to adopt.")
    elif exceeds_noise and consistent and b_delta > i_delta:
        verdict = ("SUPPORTED: fewer hops helps boundary spots more than interior spots by "
                   "more than seed noise, consistently across held-out slices -- consistent "
                   "with over-smoothing at layer boundaries.")
    else:
        reasons = []
        if not exceeds_noise:
            reasons.append(f"boundary delta ({b_delta:+.4f}) is within seed noise ({typical_noise:.4f})")
        if not consistent:
            reasons.append("per-slice boundary deltas disagree in direction")
        if default_already_best:
            reasons.append(f"the global default is already optimal on "
                           f"{default_already_best}/{len(HELD_OUT_SLICES)} held-out slices")
        verdict = ("NOT SUPPORTED: " + "; ".join(reasons) +
                   ". Do not adopt per-subject hop tuning on this evidence.")
    print(verdict)
    summary["_mechanistic"] = {
        "per_slice_boundary_deltas": per_slice_boundary,
        "typical_seed_noise": typical_noise,
        "exceeds_noise": exceeds_noise,
        "direction_consistent": consistent,
        "per_slice_best_hop": per_slice_best,
        "default_already_best_on": default_already_best,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "per_slice_per_hop": all_rows,
        "selection_slice": SELECTION_SLICE,
        "held_out_slices": HELD_OUT_SLICES,
        "selection_scores": sel_scores,
        "selected_n_hops": selected,
        "global_default_n_hops": GLOBAL_DEFAULT_HOPS,
        "held_out_summary": summary,
        "verdict": verdict,
    }, indent=2))
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
