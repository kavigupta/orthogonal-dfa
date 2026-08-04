import itertools
from typing import Optional, Tuple

import numpy as np
import scipy

#: Floor on the smaller of the prefix accept/reject rates. ``give_up_check``
#: needs it as a lower bound, and ``satisfies_preconditions`` rejects targets
#: that would break it, so both read this.
DEFAULT_MIN_ACC_REJ = 0.02


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


def row_sum_dispersion(columns) -> float:
    """Chi-square dispersion of the per-prefix row sums of ``columns`` (k x P).

    Around 1 when the prefixes are exchangeable, above 1 when they split into
    classes the columns agree about: a prefix in the accept class reads 1 more
    often in *every* column at once, so its row sum sits high and the sums
    spread wider than a single binomial.
    """
    k, num_prefixes = columns.shape
    mu = float(columns.mean())
    if mu <= 0 or mu >= 1 or num_prefixes < 2:
        return 0.0
    row_sums = columns.sum(axis=0)
    return float(
        ((row_sums - k * mu) ** 2).sum() / (k * mu * (1 - mu)) / (num_prefixes - 1)
    )


def _feasible_balance_range(empirical_pos, s, min_acc_rej):
    """Balances consistent with the observed rate.

    Fixing ``empirical_pos = p*(c+s) + (1-p)*(c-s)`` pins ``c`` once ``p`` is
    chosen, and ``c +- s`` still has to be a probability, which rules out the
    extremes: a population that is almost all one class cannot average to a
    middling rate without pushing a class rate outside ``[0, 1]``.
    """
    lo = max(min_acc_rej, 1 - (1 - empirical_pos) / (2 * s))
    hi = min(1 - min_acc_rej, empirical_pos / (2 * s))
    return (lo, hi) if lo <= hi else None


def _dispersion_at_the_bound(
    k, num_prefixes, *, empirical_pos, s, balance, quantile, num_sim
):
    """The ``quantile`` of ``row_sum_dispersion`` when a cluster does exist, at
    the given balance and the observed rate.

    Row sums are drawn straight from the two-class mixture rather than from a
    whole mask matrix, so this stays cheap as k and P grow.
    """
    accept_rate = empirical_pos + 2 * s * (1 - balance)
    reject_rate = empirical_pos - 2 * s * balance
    rng = np.random.default_rng(0)
    accept = rng.random((num_sim, num_prefixes)) < balance
    draws = rng.binomial(k, np.where(accept, accept_rate, reject_rate))
    mu = draws.mean(axis=1) / k
    ok = (mu > 0) & (mu < 1)
    if not ok.any():
        return 0.0
    draws, mu = draws[ok], mu[ok]
    spread = ((draws - (k * mu)[:, None]) ** 2).sum(axis=1)
    x2 = spread / (k * mu * (1 - mu)) / (num_prefixes - 1)
    return float(np.quantile(x2, quantile))


def give_up_check(  # pylint: disable=too-many-positional-arguments
    signal_strength,
    num_prefixes,
    num_suffixes,
    min_suffix_frequency,
    min_acc_rej,
    empirical_pos,
    *,
    failure_prob=0.01,
    num_sim=4000,
):
    """When to conclude no suffix family separates the prefixes.

    Returns ``(k, tau)``; give up if ``row_sum_dispersion`` over the top-``k``
    columns is ``<= tau``.  ``None`` when ``k < 2`` leaves nothing to test, or
    when no balance is consistent with ``empirical_pos``.

    H0 is that the cluster exists, so rejecting it is what triggers the
    destructive action and ``Pr[give up | cluster exists] <= failure_prob`` is
    the guarantee.  Under H0 a prefix's row sum is binomial in its class rate,
    a mixture whose dispersion rises with ``p * (1 - p)``.  That is monotone,
    so evaluating at the least balanced population still consistent with the
    evidence gives the lowest dispersion H0 allows, and a lower-tail threshold
    there is conservative for every better-balanced target.

    The model is pinned to ``empirical_pos`` because the statistic normalises
    by the observed rate; deriving an expected rate from anything else would
    compare the data against a threshold built for a different population.

    The statistic never reads the seed column, so the k readings in a prefix
    are independent given its class.  Selecting the columns by agreement with
    the seed only inflates dispersion, and inflation cannot trip a lower-tail
    test, so the selection needs no correction.

    :param min_acc_rej: floor on the smaller of the accept/reject rates.  A
        *bound*, not an estimate -- the guarantee holds only above it.
    :param empirical_pos: observed rate the prefixes read 1 at, over the same
        prefixes the dispersion is measured on.
    """
    assert 0 < min_suffix_frequency <= 1
    assert signal_strength > 0

    # Split the budget: k may undercount the idempotent suffixes, and the
    # threshold may sit above the true quantile.
    each = failure_prob / 2
    k = int(scipy.stats.binom.ppf(each, num_suffixes, min_suffix_frequency))
    if k < 2:
        return None
    feasible = _feasible_balance_range(empirical_pos, signal_strength, min_acc_rej)
    if feasible is None:
        return None
    lo, hi = feasible
    balance = lo if lo * (1 - lo) <= hi * (1 - hi) else hi
    tau = _dispersion_at_the_bound(
        k,
        num_prefixes,
        empirical_pos=empirical_pos,
        s=signal_strength,
        balance=balance,
        quantile=each,
        num_sim=num_sim,
    )
    return k, tau


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


def counterexample_search_exhausted(
    num_found, num_samples, count, max_samples, *, failure_prob=1e-5
):
    """Whether the hits so far are too far below ``count / max_samples``, the
    rate the search has to sustain to reach ``count``."""
    needed_rate = count / max_samples
    pval = scipy.stats.binom.cdf(num_found, num_samples, needed_rate)
    return pval < failure_prob


def binomial_side_of_boundary(num_accepts, num_samples, boundary, *, failure_prob=1e-5):
    """Binomial test of num_accepts/num_samples against accept rate ``boundary``.

    Returns True if the count is significantly above ``boundary``, False if
    significantly below, and None if neither tail clears ``failure_prob`` (including
    small ``num_samples``).
    """
    above = 1 - scipy.stats.binom.cdf(num_accepts - 1, num_samples, boundary)
    if above < failure_prob:
        return True
    below = scipy.stats.binom.cdf(num_accepts, num_samples, boundary)
    if below < failure_prob:
        return False
    return None
