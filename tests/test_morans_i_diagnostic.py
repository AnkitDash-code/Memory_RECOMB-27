import numpy as np
import scipy.sparse as sp

from src.eval.morans_i_diagnostic import morans_i


def _ring_connectivities(n):
    rows = list(range(n)) + list(range(n))
    cols = [(i + 1) % n for i in range(n)] + [(i - 1) % n for i in range(n)]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_morans_i_high_for_smooth_signal_on_ring():
    """A signal that varies smoothly around a ring (neighbors have similar
    values) should score close to +1."""
    n = 60
    conn = _ring_connectivities(n)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    smooth_signal = np.sin(theta).reshape(-1, 1)

    i = morans_i(smooth_signal, conn)

    assert i[0] > 0.8


def test_morans_i_near_zero_for_random_noise():
    rng = np.random.default_rng(0)
    n = 200
    conn = _ring_connectivities(n)
    noise = rng.normal(size=(n, 1))

    i = morans_i(noise, conn)

    assert abs(i[0]) < 0.2


def test_morans_i_negative_for_checkerboard_signal():
    """Alternating +1/-1 around a ring (every neighbor is the opposite
    value) is the textbook negative-autocorrelation case."""
    n = 60
    conn = _ring_connectivities(n)
    checkerboard = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)]).reshape(-1, 1)

    i = morans_i(checkerboard, conn)

    assert i[0] < -0.8


def test_morans_i_vectorized_across_multiple_columns():
    n = 60
    conn = _ring_connectivities(n)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    smooth = np.sin(theta)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=n)
    features = np.stack([smooth, noise], axis=1)

    i = morans_i(features, conn)

    assert i.shape == (2,)
    assert i[0] > 0.8
    assert abs(i[1]) < 0.3
