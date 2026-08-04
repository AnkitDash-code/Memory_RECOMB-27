import numpy as np
import pytest
import scipy.sparse as sp

from src.data.physical_scale import (
    PLATFORM_SPACING_UM,
    local_expression_heterogeneity,
    measure_average_edge_length_um,
    um_radius_to_hop_count,
)


def _chain(n):
    rows = list(range(n - 1)) + list(range(1, n))
    cols = list(range(1, n)) + list(range(n - 1))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_platform_spacing_table_and_radius_conversion():
    assert PLATFORM_SPACING_UM["visium"] == 55.0
    assert um_radius_to_hop_count(220, "visium", 55) == 4
    assert um_radius_to_hop_count(220, "stereoseq", 0.5) == 440


def test_radius_conversion_rejects_invalid_scale():
    with pytest.raises(ValueError):
        um_radius_to_hop_count(10, "visium", 0)
    with pytest.raises(ValueError):
        um_radius_to_hop_count(10, "not-a-platform", 1)


def test_average_edge_length_uses_measured_physical_coordinates():
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0], [60.0, 0.0]])
    length = measure_average_edge_length_um(
        coords, _chain(4), "slideseqv2", coordinates_in_um=True
    )
    # Directed chain edges are 10, 10, 20, 20, 30, 30.
    assert length == pytest.approx(20.0)


def test_local_heterogeneity_returns_physical_hop_metadata():
    features = np.array([[0.0], [0.0], [10.0], [10.0]], dtype=np.float32)
    scores, hops = local_expression_heterogeneity(
        features,
        _chain(4),
        radius_um=20,
        platform="slideseqv2",
        avg_edge_length_um=10,
    )
    assert hops == 2
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert scores[1] > 0
