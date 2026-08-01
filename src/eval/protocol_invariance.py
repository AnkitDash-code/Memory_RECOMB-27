"""Phase B1, decisive part: does the clustering-protocol handicap bias the GAP,
or only the absolute numbers?

Phase B1 established that our GraphST reproduction (0.590 on 151673) sits
~0.043 below the literature reference (0.633), and that roughly a third of
that is attributable to clustering INITIALIZATION -- R's mclust initializes
from model-based hierarchical agglomeration, sklearn's GaussianMixture
defaults to k-means. Using a Ward-agglomerative init instead lifts GraphST to
0.602.

That raises the reviewer-relevant question, which is NOT "is our absolute
number lower than the paper's". It is: **is our baseline systematically
handicapped in a way that flatters our own method?** Because the protocol is
applied identically to every method here, a protocol-level handicap should
lift or lower BOTH methods together and leave the gap roughly unchanged. If
instead the gap shrinks materially under the better protocol, then the
headline "no significant difference from GraphST" claim was partly an artifact
of a weakened baseline, and would have to be restated.

Scores BOTH methods' embeddings under BOTH initializations, on the same
slices and seeds, and compares the resulting gaps. Deliberately reuses one
embedding per (method, slice, seed) and re-clusters it two ways, so the
comparison isolates the protocol and carries no retraining variance.

Not run on all 12 slices x 5 seeds: this is a robustness check on the
protocol, not a headline number, and a 6-slice x 3-seed subset is enough to
tell "gap roughly unchanged" from "gap materially different".
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from src.data.load_dlpfc import load_dlpfc_slice
from src.data.preprocess import preprocess_hvg
from src.eval.clustering import refine_labels_spatial
from src.models.run_graphst import run_graphst
from src.models.train_spatial_address import train_spatial_address_model

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "logs" / "protocol_invariance.json"
)
# Two slices per subject, all held-out (151673 excluded as the tuning slice).
SAMPLES = ["151507", "151510", "151669", "151672", "151674", "151676"]
SEEDS = [0, 1, 2]
N_PCS = 20


def cluster(embedding, n_clusters, coords, init, seed=2020):
    reduced = PCA(n_components=min(N_PCS, embedding.shape[1]),
                  random_state=seed).fit_transform(embedding)
    kwargs = dict(n_components=n_clusters, covariance_type="tied", random_state=seed)
    if init == "hierarchical":
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(reduced)
        kwargs["means_init"] = np.stack([reduced[agg == k].mean(axis=0) for k in range(n_clusters)])
        kwargs["n_init"] = 1
    else:
        kwargs["init_params"] = init
        kwargs["n_init"] = 10
    labels = GaussianMixture(**kwargs).fit_predict(reduced).astype(str)
    return refine_labels_spatial(labels, coords, n_neighbors=50)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inits = ["kmeans", "hierarchical"]
    scores = {init: {"ours": [], "graphst": []} for init in inits}
    per_slice = []

    for sample in SAMPLES:
        raw = load_dlpfc_slice(sample)
        adata = preprocess_hvg(raw.copy())
        truth = adata.obs["ground_truth_layer"]
        mask = truth.notna().to_numpy()
        n_layers = int(truth.nunique())
        coords = adata.obsm["spatial"]
        truth_np = truth.to_numpy()

        row = {"sample": sample}
        for init in inits:
            ours, gst = [], []
            for seed in SEEDS:
                _, trained, _ = train_spatial_address_model(
                    adata.copy(), seed=seed, device=device, verbose=False
                )
                labels = cluster(trained.obsm["X_spatial_address"], n_layers, coords, init)
                ours.append(adjusted_rand_score(truth_np[mask], np.asarray(labels)[mask]))

                g = run_graphst(raw.copy(), n_clusters=n_layers, device=device,
                                random_seed=seed, cluster=False)
                gt = truth.reindex(g.obs_names)
                gmask = gt.notna().to_numpy()
                glabels = cluster(g.obsm["emb"], n_layers, g.obsm["spatial"], init)
                gst.append(adjusted_rand_score(gt.to_numpy()[gmask], np.asarray(glabels)[gmask]))

            row[init] = {"ours": float(np.mean(ours)), "graphst": float(np.mean(gst)),
                         "gap": float(np.mean(gst) - np.mean(ours))}
            scores[init]["ours"].append(np.mean(ours))
            scores[init]["graphst"].append(np.mean(gst))

        per_slice.append(row)
        print(f"{sample}: "
              + "  ".join(f"[{i}] ours={row[i]['ours']:.4f} gst={row[i]['graphst']:.4f} "
                          f"gap={row[i]['gap']:+.4f}" for i in inits), flush=True)

    print(f"\n{'init':<16}{'ours':>9}{'graphst':>10}{'gap':>9}")
    print("-" * 44)
    summary = {}
    for init in inits:
        o = float(np.mean(scores[init]["ours"]))
        g = float(np.mean(scores[init]["graphst"]))
        summary[init] = {"ours": o, "graphst": g, "gap": g - o}
        print(f"{init:<16}{o:>9.4f}{g:>10.4f}{g - o:>9.4f}")

    delta_gap = summary["hierarchical"]["gap"] - summary["kmeans"]["gap"]
    print(f"\ngap change from switching protocol: {delta_gap:+.4f}")
    if abs(delta_gap) < 0.01:
        verdict = ("PROTOCOL-INVARIANT: the gap is essentially unchanged, so the reproduction "
                   "shortfall lifts both methods together and does NOT flatter our method. "
                   "The headline comparison stands.")
    elif delta_gap > 0:
        verdict = ("CAUTION: the gap WIDENS under the better protocol -- our previous numbers "
                   "were partly flattered by a weakened baseline, and the headline claim must "
                   "be restated against the stronger protocol.")
    else:
        verdict = ("The gap NARROWS under the better protocol, i.e. the current protocol was "
                   "if anything conservative toward our method.")
    print(verdict)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "samples": SAMPLES, "seeds": SEEDS, "per_slice": per_slice,
        "summary": summary, "gap_change": delta_gap, "verdict": verdict,
    }, indent=2))
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
