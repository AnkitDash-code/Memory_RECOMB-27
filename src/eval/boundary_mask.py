"""Identify spots near an annotated layer boundary, for boundary-vs-interior
error analysis.

Phase A of the forward plan tests a specific mechanistic hypothesis: that
`n_hops=4` address propagation over-smooths genuinely separable layers on
subject 3 (which has by far the richest per-spot signal -- 2058 genes/spot vs
1324 for subject 1, see Stage 19). If that is what is happening, reducing the
hop count should specifically improve spots NEAR layer boundaries, while
leaving interior spots flat or slightly worse. An aggregate ARI bump alone
would not distinguish that mechanism from generic noise, which is why the
breakdown -- not the headline number -- is the actual test.

"Near a boundary" is defined on the spatial graph, not in pixel space: spot i
is boundary-adjacent if any spot reachable within `radius` hops of i carries a
different ground-truth layer label. Graph hops rather than Euclidean distance
because the model itself propagates over that same graph, so this measures
distance in the units the mechanism actually operates in.

Naming note: `radius` here is the boundary-definition neighbourhood and is
completely independent of the model's `n_hops` address-propagation depth.
Conflating the two would make the diagnostic circular.
"""

import numpy as np
import scipy.sparse as sp


def khop_adjacency(connectivities, radius):
    """Boolean reachability matrix within `radius` hops (excluding self)."""
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")

    adjacency = (sp.csr_matrix(connectivities) > 0).astype(bool)
    reach = adjacency.copy()
    frontier = adjacency.copy()
    for _ in range(radius - 1):
        frontier = (frontier @ adjacency).astype(bool)
        reach = (reach + frontier).astype(bool)
    reach.setdiag(False)
    reach.eliminate_zeros()
    return reach


def boundary_mask(labels, connectivities, radius=2, valid=None):
    """True for spots with a differently-labelled spot within `radius` hops.

    labels: array of ground-truth labels (object/str). Spots whose label is
    missing are excluded from both the mask and from acting as neighbours,
    since an unannotated neighbour tells us nothing about whether spot i sits
    on a real boundary -- treating NaN as "different" would spuriously mark
    tissue edges as boundaries.
    """
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    if valid is None:
        valid = np.array([l is not None and l == l for l in labels])  # NaN != NaN

    reach = khop_adjacency(connectivities, radius).tolil().rows

    out = np.zeros(n, dtype=bool)
    for i in range(n):
        if not valid[i]:
            continue
        for j in reach[i]:
            if valid[j] and labels[j] != labels[i]:
                out[i] = True
                break
    return out
