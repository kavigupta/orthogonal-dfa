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


def give_up_check(
    signal_strength,
    num_prefixes,
    num_suffixes,
    min_suffix_frequency,
    min_acc_rej,
    empirical_pos,
    failure_prob=0.01,
):
    """
    Checks whether we should give up on finding suffixes based on the parameters of the problem and the empirical agreement with the seed.

    :param signal_strength: The minimum signal strength we are trying to detect.
    :param num_prefixes: The number of prefixes we have.
    :param num_suffixes: The number of suffixes we have sampled so far.
    :param min_suffix_frequency: The minimum frequency of suffixes we are trying to find.
    :param min_acc_rej: The minimum of the accept and reject rates of the prefixes.
    :param empirical_pos: The empirical positive rate of the prefixes.
    :param failure_prob: The probability with which we are willing to fail to find a good suffix.
    :return: A tuple of (k, agreement_threshold). We should give up if ``mean([mask[i] == mask[0] for i in top_k]) < agreement_threshold``.
    """
    r = min_suffix_frequency
    s = signal_strength
    P = num_prefixes
    T = num_suffixes
    assert 0 < r <= 1
    assert s > 0

    # k = lower bound on number of idempotent suffixes.
    # We can assume that the top k are idempotent, because the ones
    # that aren't are better than a random idempotent suffix anyway,
    # so this is a conservative threshold.
    k = int(scipy.stats.binom.ppf(failure_prob / 3, T, r))
    if k < 2:
        return None

    # We now know we have at least k idempotent suffixes, but these are noisy.
    # Specifically, each has v_{ij} ~ B(center + s) if prefix i is accept, and v_{ij} ~ B(center - s) if prefix i is reject.
    # This means if we let w_{ij} = v_{ij} == v_{0j} (agreement with seed),
    #    we have that w_{ij} = 1 iff either
    #        - prefix i is accept and two samples from B(center + s) agree
    #        - prefix i is reject and two samples from B(center - s) agree
    # The probability that two samples from B(p) agree is a(p) = p^2 + (1-p)^2 = 2p^2 - 2p + 1 = 1 - 2p(1-p).
    # This is a quadratic with minimum at p=0.5, where it equals 0.5, and it increases as p goes to 0 or 1.
    # As such, we have that w_{ij} = 1 with probability
    #       p_same
    #           = p_acc * a(c + s) + (1 - p_acc) * a(c - s)
    #           = p_acc * (1 - 2(c + s)(1 - (c + s))) + (1 - p_acc) * (1 - 2(c - s)(1 - (c - s)))
    #           = 1 - 2 * p_acc * ((c + s)(1 - (c + s)) - (c - s)(1 - (c - s)))
    # One thing is that we know the quantity of empriical positives the oracle on the prefixes:
    #       empirical_pos = p_acc * (c + s) + (1 - p_acc) * (c - s)
    # So if we subtract out the expected agreement based on the empirical positive rate, we get
    #       p_same - a(empirical_pos)
    # Which we can simplify (see _give_up_check_sym) to
    #       8*p_acc (1 - p_acc) s^2
    # This is minimized when p_acc is minimized
    # We can thus compute
    def a(p):
        return 1 - 2 * p * (1 - p)

    p_same = 8 * min_acc_rej * (1 - min_acc_rej) * s**2 + a(empirical_pos)

    # We want to give up if the top-k mean agreement is less than p_same, which is the expected agreement of a random idempotent suffix.
    # The top-k mean agreement can be computed as
    # sum_ij w_{ij} / (kP)
    # Which is just a binomial distribution with kP trials and probability p_same, divided by kP.
    agreement_threshold = scipy.stats.binom.ppf(1 - failure_prob / 3, k * P, p_same) / (
        k * P
    )
    return k, agreement_threshold


def _give_up_check_sym():
    import sympy

    center = sympy.Symbol("center")
    s = sympy.Symbol("s")
    p_acc = sympy.Symbol("p_acc")
    a = lambda p: 1 - 2 * p * (1 - p)
    expr = p_acc * a(center + s) + (1 - p_acc) * a(center - s)
    print("correlation", sympy.simplify(expr))
    empirical_pos = p_acc * (center + s) + (1 - p_acc) * (center - s)
    expected = a(empirical_pos)
    print("expected", sympy.simplify(expected))
    delta = sympy.simplify(expr - expected)
    print("delta", delta)


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
