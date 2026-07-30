"""Count-based reconstruction likelihoods for zero-inflated spatial transcriptomics.

Measured sparsity of the datasets used here is 68-97% zeros
(`outputs/logs/data_stats.txt`). Under an MSE objective on scaled expression,
those zeros are treated as ordinary real values with Gaussian noise, which is
the wrong likelihood: a zero can arise either because the gene is genuinely not
expressed or because of a dropout event, and the two carry different
information.

Negative binomial (NB) models overdispersed counts; zero-inflated negative
binomial (ZINB) adds an explicit dropout probability on top. This is the same
modelling choice stGRL cites as its core contribution for exactly this reason.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def nb_loss(x, mu, theta, reduction="mean"):
    """Negative binomial negative log-likelihood.

    x     : observed counts (non-negative)
    mu    : predicted mean (positive)
    theta : inverse-dispersion (positive); large theta -> Poisson limit
    """
    mu = mu + EPS
    theta = theta + EPS

    log_theta_mu = torch.log(theta + mu)
    # Build the LOG-LIKELIHOOD first and negate once at the end; writing the
    # NLL term-by-term makes it easy to drop a sign (an earlier version did,
    # caught by the scipy reference test).
    log_likelihood = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mu) - log_theta_mu)
    )
    nll = -log_likelihood
    return nll.mean() if reduction == "mean" else nll.sum()


def zinb_loss(x, mu, theta, pi_logits, reduction="mean"):
    """Zero-inflated negative binomial negative log-likelihood.

    pi_logits : logits of the zero-inflation (dropout) probability.

    Computed in log-space via softplus/logsumexp rather than by exponentiating
    probabilities directly -- the naive form underflows badly when mu is small,
    which is the common case here given how sparse the data is.
    """
    mu = mu + EPS
    theta = theta + EPS

    softplus_pi = F.softplus(-pi_logits)
    log_theta_mu = torch.log(theta + mu)
    log_theta_frac = theta * (torch.log(theta) - log_theta_mu)

    # log P(x = 0) under the NB component, mixed with the dropout point mass.
    zero_case = F.softplus(-pi_logits + log_theta_frac) - softplus_pi

    # log P(x = k > 0): must include log(1 - pi), i.e. -softplus(pi_logits).
    nonzero_case = (
        -softplus_pi
        - pi_logits
        + log_theta_frac
        + x * (torch.log(mu) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )

    log_likelihood = torch.where(x < EPS, zero_case, nonzero_case)
    nll = -log_likelihood
    return nll.mean() if reduction == "mean" else nll.sum()


class CountDecoder(nn.Module):
    """Decode an embedding into NB / ZINB parameters over genes.

    Predicts the mean as a proportion of each spot's own library size, which is
    the standard scVI-style parameterization: it keeps the model from having to
    re-learn per-spot sequencing depth, which is technical variation rather than
    biology.
    """

    def __init__(self, embedding_dim, n_genes, hidden_dim=256, zero_inflated=True):
        super().__init__()
        self.zero_inflated = zero_inflated
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, n_genes)
        # Per-gene dispersion, shared across spots (as in scVI's default).
        self.log_theta = nn.Parameter(torch.randn(n_genes) * 0.1)
        self.dropout_head = nn.Linear(hidden_dim, n_genes) if zero_inflated else None

    def forward(self, embedding, library_size):
        hidden = self.shared(embedding)
        mean_proportion = F.softmax(self.mean_head(hidden), dim=-1)
        mu = mean_proportion * library_size
        theta = torch.exp(self.log_theta).clamp(max=1e6)
        pi_logits = self.dropout_head(hidden) if self.zero_inflated else None
        return mu, theta, pi_logits
