"""Phase D diagnosis: why does Phase C's breast cancer result show a real gap
to GraphST while DLPFC shows near-parity?

Compares ground-truth domain size against the model's own k-hop propagation
reach, on both datasets, using the identical preprocessing/graph-construction
pipeline (`preprocess_hvg`) each is actually trained on.

Finding: DLPFC's 7 layers average 623 spots (min 166 across all 12 slices) --
always larger than the ~55-60 spots reachable within `n_hops=4` (the
cross-validated default, chosen ON DLPFC). Breast cancer's 20
pathologist-annotated regions average only 190 spots, several as small as
28-53 -- SMALLER than a single 4-hop neighbourhood. A fixed global hop count
tuned where it never exceeds domain size will, by construction, over-smooth
domains where it does. See `src/models/memory_layer.py`'s
`SpatialAddressMemoryLayer` docstring for how this motivated an architectural
fix attempt (adaptive_hops), and `cross_validate_adaptive_hops.py` for why
that attempt did not pan out.

A second candidate explanation -- that expression-weighted adjacency (the
existing boundary-blur safeguard, Stage 13) discriminates domain boundaries
worse on breast cancer than DLPFC -- was checked here too and RULED OUT: the
diff/same edge-weight ratio is actually LOWER (better separation) on breast
cancer (0.862) than DLPFC (0.937), so edge-weight quality is not the
bottleneck; domain SIZE relative to propagation depth is.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from src.data.load_breast_cancer import load_breast_cancer
from src.data.load_dlpfc import ALL_DLPFC_SAMPLES, load_dlpfc_slice
from src.data.preprocess import get_hvg_features, preprocess_hvg
from src.eval.boundary_mask import khop_adjacency
from src.models.memory_layer import expression_weighted_adjacency

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "domain_scale_diagnostic.json"


def domain_sizes(adata, truth_col):
    truth = adata.obs[truth_col].dropna()
    return truth.value_counts().to_numpy()


def domain_fragmentation(adata, truth_col, coord_type="grid"):
    """Connected components per domain in the spatial graph -- large values
    mean a domain is spatially fragmented (multiple disjoint patches) rather
    than one contiguous region."""
    adata = preprocess_hvg(adata.copy(), coord_type=coord_type)
    truth = adata.obs[truth_col]
    conn = adata.obsp["spatial_connectivities"]
    out = {}
    for domain in truth.dropna().unique():
        idx = np.where(truth.to_numpy() == domain)[0]
        if len(idx) < 2:
            out[domain] = (len(idx), 1)
            continue
        sub = conn[idx][:, idx]
        n_components, _ = connected_components(sub, directed=False)
        out[domain] = (len(idx), n_components)
    return out


def hop_reach_stats(adata, coord_type, n_hops=4):
    adata = preprocess_hvg(adata.copy(), coord_type=coord_type)
    conn = adata.obsp["spatial_connectivities"]
    khop = khop_adjacency(conn, n_hops)
    return np.asarray((khop > 0).sum(axis=1)).flatten()


def boundary_edge_weight_stats(adata, truth_col, coord_type="grid", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = preprocess_hvg(adata.copy(), coord_type=coord_type)
    truth = adata.obs[truth_col].to_numpy()
    hvg = get_hvg_features(adata)
    adj = expression_weighted_adjacency(adata.obsp["spatial_connectivities"], hvg, device=device).coalesce()
    row, col = adj.indices().cpu().numpy()
    vals = adj.values().cpu().numpy()
    same = truth[row] == truth[col]
    return {
        "same_domain_mean_weight": float(vals[same].mean()),
        "diff_domain_mean_weight": float(vals[~same].mean()),
        "diff_over_same_ratio": float(vals[~same].mean() / vals[same].mean()),
    }


def main():
    print("=== Domain size: DLPFC (all 12 slices) vs breast cancer ===")
    dlpfc_sizes = []
    for sample in ALL_DLPFC_SAMPLES:
        dlpfc_sizes.extend(domain_sizes(load_dlpfc_slice(sample), "ground_truth_layer").tolist())
    dlpfc_sizes = np.array(dlpfc_sizes)

    bc_raw = load_breast_cancer()
    bc_sizes = domain_sizes(bc_raw, "ground_truth_region")

    print(f"DLPFC:         mean={dlpfc_sizes.mean():.1f} median={np.median(dlpfc_sizes):.1f} "
          f"min={dlpfc_sizes.min()} max={dlpfc_sizes.max()}")
    print(f"Breast cancer: mean={bc_sizes.mean():.1f} median={np.median(bc_sizes):.1f} "
          f"min={bc_sizes.min()} max={bc_sizes.max()}")

    print("\n=== 4-hop reachable-neighbour counts (151673 vs breast cancer) ===")
    reach_dlpfc = hop_reach_stats(load_dlpfc_slice("151673"), "grid", n_hops=4)
    reach_bc = hop_reach_stats(bc_raw, "grid", n_hops=4)
    print(f"DLPFC 151673:  mean={reach_dlpfc.mean():.1f} median={np.median(reach_dlpfc):.1f}")
    print(f"Breast cancer: mean={reach_bc.mean():.1f} median={np.median(reach_bc):.1f}")
    n_bc_below = int((bc_sizes < np.median(reach_bc)).sum())
    print(f"-> {n_bc_below}/{len(bc_sizes)} breast cancer domains are SMALLER than "
          f"the median 4-hop reachable-neighbour count")

    print("\n=== Domain fragmentation (connected components) ===")
    frag_dlpfc = domain_fragmentation(load_dlpfc_slice("151673"), "ground_truth_layer")
    frag_bc = domain_fragmentation(bc_raw, "ground_truth_region")
    print("DLPFC 151673 mean n_components:", np.mean([n for _, n in frag_dlpfc.values()]))
    print("Breast cancer mean n_components:", np.mean([n for _, n in frag_bc.values()]))

    print("\n=== Expression-weighted edge separation at domain boundaries ===")
    edge_dlpfc = boundary_edge_weight_stats(load_dlpfc_slice("151673"), "ground_truth_layer")
    edge_bc = boundary_edge_weight_stats(bc_raw, "ground_truth_region")
    print("DLPFC 151673:", edge_dlpfc)
    print("Breast cancer:", edge_bc)
    print("(lower diff_over_same_ratio = better boundary separation; "
          "breast cancer's is LOWER, ruling out 'worse edge weighting' as the explanation)")

    results = {
        "domain_sizes": {
            "dlpfc": {"mean": float(dlpfc_sizes.mean()), "median": float(np.median(dlpfc_sizes)),
                      "min": int(dlpfc_sizes.min()), "max": int(dlpfc_sizes.max())},
            "breast_cancer": {"mean": float(bc_sizes.mean()), "median": float(np.median(bc_sizes)),
                               "min": int(bc_sizes.min()), "max": int(bc_sizes.max())},
        },
        "hop_reach_4hop": {
            "dlpfc_151673": {"mean": float(reach_dlpfc.mean()), "median": float(np.median(reach_dlpfc))},
            "breast_cancer": {"mean": float(reach_bc.mean()), "median": float(np.median(reach_bc))},
            "n_bc_domains_smaller_than_median_reach": n_bc_below,
            "n_bc_domains_total": len(bc_sizes),
        },
        "fragmentation": {
            "dlpfc_151673_mean_n_components": float(np.mean([n for _, n in frag_dlpfc.values()])),
            "breast_cancer_mean_n_components": float(np.mean([n for _, n in frag_bc.values()])),
        },
        "boundary_edge_weights": {"dlpfc_151673": edge_dlpfc, "breast_cancer": edge_bc},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
