import numpy as np
import pytest
import scipy.sparse as sp

from src.eval.boundary_mask import boundary_mask, khop_adjacency


def _chain(n):
    """Path graph 0-1-2-...-(n-1)."""
    rows = list(range(n - 1)) + list(range(1, n))
    cols = list(range(1, n)) + list(range(n - 1))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_khop_adjacency_radius_one_is_the_graph_itself():
    conn = _chain(5)
    reach = khop_adjacency(conn, 1).toarray()

    assert reach[0, 1] and not reach[0, 2]
    assert not reach[0, 0]  # no self


def test_khop_adjacency_radius_two_reaches_two_steps():
    conn = _chain(5)
    reach = khop_adjacency(conn, 2).toarray()

    assert reach[0, 1] and reach[0, 2]
    assert not reach[0, 3]
    assert not reach[0, 0]


def test_khop_adjacency_rejects_zero_radius():
    with pytest.raises(ValueError):
        khop_adjacency(_chain(3), 0)


def test_boundary_mask_flags_spots_adjacent_to_a_different_label():
    # labels: A A A | B B B  -- boundary sits between index 2 and 3
    conn = _chain(6)
    labels = np.array(["A", "A", "A", "B", "B", "B"], dtype=object)

    mask = boundary_mask(labels, conn, radius=1)

    assert list(mask) == [False, False, True, True, False, False]


def test_boundary_mask_radius_two_widens_the_band():
    conn = _chain(6)
    labels = np.array(["A", "A", "A", "B", "B", "B"], dtype=object)

    mask = boundary_mask(labels, conn, radius=2)

    assert list(mask) == [False, True, True, True, True, False]


def test_boundary_mask_all_false_when_every_label_identical():
    conn = _chain(6)
    labels = np.array(["A"] * 6, dtype=object)

    assert not boundary_mask(labels, conn, radius=2).any()


def test_boundary_mask_ignores_unannotated_neighbours():
    """A NaN-labelled neighbour must not count as 'different' -- otherwise
    tissue edges next to unannotated spots get spuriously marked as boundaries."""
    conn = _chain(4)
    labels = np.array(["A", "A", np.nan, "A"], dtype=object)

    mask = boundary_mask(labels, conn, radius=1)

    # No spot has a differently-*annotated* neighbour, so nothing is boundary.
    assert not mask.any()
    # And the unannotated spot itself is never marked.
    assert not mask[2]
