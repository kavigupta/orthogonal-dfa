import itertools
from typing import Optional, Tuple

import numpy as np
import scipy


def population_size_and_evidence_margin(
    signal_strength, acceptable_fpr, acceptable_fnr
) -> Tuple[int, float]:
    """
    Decisions will be made by taking N samples and seeing if the proportion is outside
    (center - epsilon, center + epsilon). The true distribution has accept rate
    center + signal_strength and reject rate center - signal_strength.

    We want FPR (under the null B(center)) at most acceptable_fpr, and FNR (under
    the true distribution) at most acceptable_fnr.
    """
    assert signal_strength > 0
    N_low = 1
    N_high = None
    while N_high is None or N_low < N_high:
        if N_high is None:
            N_try = N_low * 2
        else:
            N_try = (N_low + N_high) // 2
        result = evidence_margin_for_population_size(
            signal_strength, acceptable_fpr, acceptable_fnr, N_try
        )
        if result is None:
            N_low = N_try + 1
        else:
            N_high = N_try
    res = evidence_margin_for_population_size(
        signal_strength, acceptable_fpr, acceptable_fnr, N_high
    )
    assert res is not None
    return res


def evidence_margin_for_population_size(
    signal_strength, acceptable_fpr, acceptable_fnr, N, *, center=0.5
) -> Optional[Tuple[int, float]]:
    """
    See population_size_and_evidence_margin for context.
    """
    for eps in np.linspace(0.01, signal_strength, 100):
        k_low = int(np.floor(N * (center - eps)))
        k_high = int(np.ceil(N * (center + eps)))
        fpr = scipy.stats.binom.cdf(k_low, N, center) + (
            1 - scipy.stats.binom.cdf(k_high - 1, N, center)
        )
        fnr = scipy.stats.binom.cdf(
            k_high - 1, N, signal_strength + center
        ) - scipy.stats.binom.cdf(k_low, N, signal_strength + center)
        if fpr <= acceptable_fpr and fnr <= acceptable_fnr:
            return N, eps
    return None


def compute_prefix_set_size(delta, noise_level, acceptable_misclassification):
    r"""
    Computes the required number of prefixes to achieve a desired misclassification rate
    when finding suffixes.

    We conceptualize the process of finding suffixes as follows:

        We have a distribution V over binary strings $2^k$ defined as

        x <- X; v_i <- x_i \oplus B(p)

        We have access to one

        v_0 ~ P(v | x = x_0)

        I want to find a set of elements from P(v | x = x_0) but can only sample from V

    where p is the noise level, and k is the quantity we want to find.

    If we look at hamming distance, we have if v ~ P(v | X=x) that, letting n be the noise vector XORd with the x,
    we have

    d(v, v_0)
        := sum_j 1(v[j] ≠ v_0[j])
        := sum_j 1(x[j] ⊕ n[j] ≠ x_0[j] ⊕ n_0[j])
        := sum_j 1(x[j] ≠ x_0[j]) ⊕ 1(n[j] ≠ n_0[j])

    The distribution (n_0[j] ≠ n[j]) is Bernoulli with parameter 2p(1-p). Let r = 2p(1-p).

    Let A = d(x, x_0) and B = k - A

    Then, we can split

    d(v, v_0)
        = sum_{j: x[j] = x_0[j]} 1(n[j] ≠ n_0[j]) + sum_{j: x[j] ≠ x_0[j]} 1(n[j] = n_0[j])
        = Binomial(B, r) + A - Binomial(A, r)
        ~= A + Normal(B * r, B * r * (1 - r)) - Normal(A * r, A * r * (1 - r))
        = A + Normal((B - A) * r, (A + B) * r * (1 - r))
        = A + Normal((k - 2A) * r, k * r * (1 - r))

    Let delta = A/k. Then, we have
    d(v, v_0)/k = delta + Normal((1 - 2 delta) * r, r * (1 - r) / k)
                = delta + r - 2 * delta * r + Normal(0, r * (1 - r) / k)

    We want to bound the probability that d(v', v_0)/k < d(v'', v_0)/k for v' ~ P(v | x = x_0) and v'' ~ P(v | x ≠ x_0).

    d(v', v_0)/k > d(v'', v_0)/k
    r + Normal(0, r * (1 - r) / k) > delta + r - 2 * delta * r + Normal(0, r * (1 - r) / k)
    Normal(0, r * (1 - r) / k) > delta - 2 * delta * r + Normal(0, r * (1 - r) / k)
    Normal(0, r * (1 - r) / k) > delta * (1 - 2 * r) + Normal(0, r * (1 - r) / k)
    Normal(0, 2 * r * (1 - r) / k) > delta * (1 - 2 * r)
    Normal(0, 1) > delta * (1 - 2 * r) / sqrt(2 * r * (1 - r) / k)

    Letting z = Φ^{-1}(1 - acceptable_misclassification), we want

    delta * (1 - 2 * r) / sqrt(2 * r * (1 - r) / k) = z
    delta^2 * (1 - 2 * r)^2 k / (2 * r * (1 - r)) = z^2
    k  = z^2 (2 * r * (1 - r))  / (delta^2 * (1 - 2 * r)^2)
    """
    r = 2 * noise_level * (1 - noise_level)
    z = scipy.stats.norm.ppf(1 - acceptable_misclassification)
    k = (z**2 * 2 * r * (1 - r)) / (delta**2 * (1 - 2 * r) ** 2)
    return int(np.ceil(k))


def compute_suffix_size_counterexample_gen(acceptable_misclassification, noise_level):
    """
    Computes the suffix size to use for counterexample generation.
    This is an alias for compute_suffix_size_for_counterexample_generation
    to match the naming convention of other hyperparameter generators.
    """
    for n in itertools.count(start=1):
        if scipy.stats.binom.cdf(n // 2, n, noise_level) < acceptable_misclassification:
            return n
    raise ValueError("not reachable")
