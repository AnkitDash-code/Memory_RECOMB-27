"""Physical-scale utilities for cross-platform spatial graphs.

The model operates on graph hops, but a hop is not a portable unit: one hop
on a Visium slide and one hop on a bead-based platform cover very different
distances.  This module keeps the conversion in one place and records the
measured graph scale on preprocessed ``AnnData`` objects.

Coordinates in many Visium files are image pixels rather than micrometres.
When coordinates are not explicitly marked as micrometres, the nominal
platform spacing is used as the unit calibration and the *shape* of the
measured graph determines the slide-specific mean edge length.  Callers that
have a physical coordinate scale can pass ``native_scale_um`` (or mark the
coordinates as micrometres) to avoid that calibration.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.sparse as sp


PLATFORM_SPACING_UM = {
    "visium": 55.0,
    "slideseqv2": 10.0,
    "stereoseq": 0.5,
}

_PLATFORM_ALIASES = {
    "10x": "visium",
    "10x_visium": "visium",
    "stereo": "stereoseq",
    "stereo-seq": "stereoseq",
    "stereo_seq": "stereoseq",
    "slide": "slideseqv2",
    "slide-seqv2": "slideseqv2",
    "slide_seq_v2": "slideseqv2",
}


def canonical_platform(platform: str) -> str:
    """Return the canonical platform key used by this module."""
    if not isinstance(platform, str) or not platform.strip():
        raise ValueError("platform must be a non-empty string")
    key = platform.strip().lower()
    key = _PLATFORM_ALIASES.get(key, key)
    if key not in PLATFORM_SPACING_UM:
        known = ", ".join(sorted(PLATFORM_SPACING_UM))
        raise ValueError(f"Unknown platform {platform!r}; expected one of: {known}")
    return key


def um_radius_to_hop_count(
    radius_um: float,
    platform: str,
    avg_edge_length_um: float,
) -> int:
    """Convert a physical radius to a graph-hop radius for one dataset.

    ``avg_edge_length_um`` must be measured from that dataset's spatial graph;
    it is deliberately not inferred from the platform table.  The platform
    argument validates the physical-scale namespace and keeps call sites
    explicit, while the measured edge length controls the conversion.
    """
    canonical_platform(platform)
    radius_um = float(radius_um)
    avg_edge_length_um = float(avg_edge_length_um)
    if not math.isfinite(radius_um) or radius_um <= 0:
        raise ValueError(f"radius_um must be finite and > 0, got {radius_um!r}")
    if not math.isfinite(avg_edge_length_um) or avg_edge_length_um <= 0:
        raise ValueError(
            "avg_edge_length_um must be finite and > 0, "
            f"got {avg_edge_length_um!r}"
        )
    return max(1, int(round(radius_um / avg_edge_length_um)))


def _graph_edges(connectivities: Any) -> tuple[np.ndarray, np.ndarray]:
    graph = sp.coo_matrix(connectivities)
    mask = np.isfinite(graph.data) & (graph.data > 0) & (graph.row != graph.col)
    rows = graph.row[mask].astype(np.int64, copy=False)
    cols = graph.col[mask].astype(np.int64, copy=False)
    if rows.size == 0:
        raise ValueError("spatial graph has no positive off-diagonal edges")
    return rows, cols


def measure_average_edge_length_um(
    coords: np.ndarray,
    connectivities: Any,
    platform: str,
    *,
    coordinates_in_um: bool = False,
    native_scale_um: float | None = None,
) -> float:
    """Measure the mean positive graph-edge length in micrometres.

    The graph topology, rather than an assumed nearest-neighbour count, is
    used to select the edges.  If coordinates are in micrometres, their raw
    Euclidean lengths are returned.  Otherwise ``native_scale_um`` can supply
    a known unit conversion.  With no explicit conversion, the median measured
    edge is calibrated to the platform's nominal spot/bead spacing; the mean
    then retains slide-specific variation in the graph geometry.
    """
    platform = canonical_platform(platform)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must have shape (n_observations, >=2)")
    rows, cols = _graph_edges(connectivities)
    if rows.max(initial=-1) >= len(coords) or cols.max(initial=-1) >= len(coords):
        raise ValueError("spatial graph contains an index outside coords")

    native_lengths = np.linalg.norm(coords[rows] - coords[cols], axis=1)
    native_lengths = native_lengths[np.isfinite(native_lengths) & (native_lengths > 0)]
    if native_lengths.size == 0:
        raise ValueError("spatial graph edges have no positive finite lengths")

    if coordinates_in_um:
        scale = 1.0
    elif native_scale_um is not None:
        scale = float(native_scale_um)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"native_scale_um must be finite and > 0, got {scale!r}")
    else:
        # A grid graph often stores unit-spaced coordinates while Visium files
        # may store image pixels.  Calibrate the measured native median to the
        # platform's nominal physical spacing, then retain the measured mean.
        scale = PLATFORM_SPACING_UM[platform] / float(np.median(native_lengths))
    return float(np.mean(native_lengths) * scale)


def annotate_spatial_scale(
    adata,
    platform: str,
    *,
    coordinates_in_um: bool = False,
    native_scale_um: float | None = None,
) -> dict[str, Any]:
    """Measure and store graph scale metadata on a preprocessed AnnData."""
    if "spatial" not in adata.obsm:
        raise KeyError("AnnData must contain obsm['spatial'] before scale measurement")
    platform = canonical_platform(platform)
    edge_length_um = measure_average_edge_length_um(
        adata.obsm["spatial"],
        adata.obsp["spatial_connectivities"],
        platform,
        coordinates_in_um=coordinates_in_um,
        native_scale_um=native_scale_um,
    )
    metadata = {
        "platform": platform,
        "nominal_spacing_um": PLATFORM_SPACING_UM[platform],
        "average_edge_length_um": edge_length_um,
        "coordinates_in_um": bool(coordinates_in_um),
        "native_scale_um": native_scale_um,
    }
    adata.uns["spatial_scale"] = metadata
    return metadata


def get_average_edge_length_um(adata, platform: str | None = None) -> float:
    """Read stored graph scale metadata, measuring it if necessary."""
    metadata = adata.uns.get("spatial_scale")
    if metadata is not None:
        if platform is not None and canonical_platform(platform) != metadata["platform"]:
            raise ValueError(
                f"AnnData scale metadata is for {metadata['platform']!r}, "
                f"not requested platform {platform!r}"
            )
        return float(metadata["average_edge_length_um"])
    platform = canonical_platform(platform or "visium")
    metadata = annotate_spatial_scale(adata, platform)
    return float(metadata["average_edge_length_um"])


def _row_normalized_without_self(connectivities):
    graph = (sp.csr_matrix(connectivities) > 0).astype(np.float32)
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return sp.diags(1.0 / degree) @ graph


def local_expression_heterogeneity(
    features: np.ndarray,
    connectivities,
    radius_um: float,
    platform: str,
    avg_edge_length_um: float,
) -> tuple[np.ndarray, int]:
    """Compute one local expression-dissimilarity score per observation.

    The radius is converted to the dataset's graph-hop radius here, so the
    heterogeneity proxy cannot silently inherit a Visium-specific hop count.
    At each reachable depth up to 32, the score averages the squared distance
    from a spot to its depth-specific graph neighbourhood mean.  For larger
    radii, a single exact ``P**h`` diffusion endpoint is used to keep
    subcellular graphs tractable while still respecting the requested maximum
    hop distance.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape (n_observations, n_features)")
    graph = _row_normalized_without_self(connectivities)
    if graph.shape[0] != features.shape[0]:
        raise ValueError("features and connectivities must have the same number of observations")
    hops = um_radius_to_hop_count(radius_um, platform, avg_edge_length_um)

    # Feature-wise scaling keeps the proxy from being dominated by a few
    # high-variance HVGs while preserving local expression dissimilarity.
    scale = np.std(features, axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    scaled = features / scale

    if hops <= 32:
        current = scaled
        scores = np.zeros(features.shape[0], dtype=np.float64)
        for _ in range(hops):
            current = graph @ current
            scores += np.mean((scaled - current) ** 2, axis=1, dtype=np.float64)
        scores /= hops
    else:
        # Sparse matrix exponentiation uses O(log hops) sparse multiplies and
        # avoids materializing a dense all-pairs radius graph.
        exponent = hops
        power = graph.tocsr()
        identity = sp.eye(graph.shape[0], format="csr", dtype=np.float32)
        while exponent:
            if exponent & 1:
                identity = identity @ power
            exponent >>= 1
            if exponent:
                power = power @ power
        current = identity @ scaled
        scores = np.mean((scaled - current) ** 2, axis=1, dtype=np.float64)

    scores = np.asarray(scores, dtype=np.float32)
    scores[~np.isfinite(scores)] = 0.0
    return scores, hops
