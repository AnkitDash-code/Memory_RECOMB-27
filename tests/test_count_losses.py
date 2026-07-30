import math

import torch

from src.models.count_losses import CountDecoder, nb_loss, zinb_loss


def test_nb_matches_scipy_reference():
    """NB NLL must match scipy's nbinom pmf under the (n=theta, p=theta/(theta+mu))
    parameterization -- the likelihood math is easy to get subtly wrong."""
    from scipy.stats import nbinom

    x = torch.tensor([0.0, 1.0, 5.0, 20.0])
    mu = torch.tensor([2.0, 3.0, 4.0, 10.0])
    theta = torch.tensor([1.5, 2.0, 5.0, 3.0])

    ours = nb_loss(x, mu, theta, reduction="none" if False else "mean")

    p = (theta / (theta + mu)).numpy()
    expected = -nbinom.logpmf(x.numpy(), n=theta.numpy(), p=p).mean()

    assert math.isclose(ours.item(), float(expected), rel_tol=1e-5)


def test_zinb_reduces_to_nb_when_dropout_probability_is_zero():
    """pi -> 0 (very negative logits) must recover the plain NB likelihood."""
    x = torch.tensor([0.0, 2.0, 7.0])
    mu = torch.tensor([1.0, 3.0, 5.0])
    theta = torch.tensor([2.0, 2.0, 2.0])
    pi_logits = torch.full_like(x, -30.0)

    assert torch.allclose(
        zinb_loss(x, mu, theta, pi_logits), nb_loss(x, mu, theta), atol=1e-4
    )


def test_zinb_prefers_dropout_when_zeros_are_excess():
    """With far more zeros than the NB mean explains, allowing dropout must give
    a strictly better (lower) NLL than forcing pi=0."""
    x = torch.zeros(100)
    x[:5] = 4.0
    mu = torch.full((100,), 4.0)
    theta = torch.full((100,), 5.0)

    with_dropout = zinb_loss(x, mu, theta, torch.full((100,), 2.0))
    without_dropout = zinb_loss(x, mu, theta, torch.full((100,), -30.0))

    assert with_dropout < without_dropout


def test_losses_are_finite_on_extreme_sparsity():
    """Guard the underflow path: tiny mu with all-zero counts is the common case
    in 97%-sparse data and must not produce NaN/Inf."""
    x = torch.zeros(64)
    mu = torch.full((64,), 1e-6)
    theta = torch.full((64,), 1e-3)
    pi_logits = torch.zeros(64)

    assert torch.isfinite(nb_loss(x, mu, theta))
    assert torch.isfinite(zinb_loss(x, mu, theta, pi_logits))


def test_count_decoder_respects_library_size():
    n_spots, n_genes, embedding_dim = 12, 30, 8
    decoder = CountDecoder(embedding_dim, n_genes)
    embedding = torch.randn(n_spots, embedding_dim)
    library_size = torch.full((n_spots, 1), 1000.0)

    mu, theta, pi_logits = decoder(embedding, library_size)

    assert mu.shape == (n_spots, n_genes)
    assert theta.shape == (n_genes,)
    assert pi_logits.shape == (n_spots, n_genes)
    assert (mu > 0).all()
    assert (theta > 0).all()
    # Mean is a proportion of library size, so per-spot totals recover it.
    assert torch.allclose(mu.sum(dim=-1), library_size.squeeze(-1), rtol=1e-4)


def test_count_decoder_can_disable_zero_inflation():
    decoder = CountDecoder(8, 20, zero_inflated=False)
    mu, theta, pi_logits = decoder(torch.randn(5, 8), torch.full((5, 1), 500.0))

    assert pi_logits is None
    assert mu.shape == (5, 20)
