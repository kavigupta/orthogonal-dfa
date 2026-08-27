import itertools
import math
from typing import Iterator, Optional, Tuple

import numpy as np
import scipy
import scipy.special


def _binom_cdf(k, n, p):
    """``scipy.stats.binom.cdf(k, n, p)``, ~30x faster on scalars.

    The generic ``rv_discrete.cdf`` broadcasts and masks its arguments on every
    call, which dominates the hot statistical tests here. ``bdtr`` is the same
    function without that; it returns nan rather than clamping off-support ``k``.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return scipy.special.bdtr(k, n, p)


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


def candidate_tests(N: int, center: float) -> Iterator[Tuple[int, int, float]]:
    """Every test over N samples, ascending in margin, as (k_low, k_high, eps):
    reject at counts <= k_low, accept at counts >= k_high, undecided between.

    The runtime spends the margin as `count / N` against `center +/- eps`, so it can
    only name a band whose ends straddle N * center to within one count:

        |2 * N * center - (k_low + k_high)| < 1

    Each such band is named by a whole interval of margins, and eps is its midpoint.
    """
    two_a = 2 * N * center
    floor = math.floor(two_a)
    for width in itertools.count(1):
        ends = floor + (width - floor) % 2
        if not two_a - 1 < ends < two_a + 1:
            continue
        k_low, k_high = (ends - width) // 2, (ends + width) // 2
        if k_low < 0 or k_high > N:
            return
        eps = (width - 1) / (2 * N)
        # An offset within float error of 1 passes the test above on an interval
        # too narrow to hold any margin. It takes a center that is a near-exact
        # rational over 2N, so it is rare and silent rather than loud.
        if k_low / N < center - eps <= (k_low + 1) / N and (
            (k_high - 1) / N < center + eps <= k_high / N
        ):
            yield k_low, k_high, eps


def evidence_margin_for_population_size(
    signal_strength, acceptable_fpr, acceptable_fnr, N, *, center=0.5
) -> Optional[Tuple[int, float]]:
    """
    See population_size_and_evidence_margin for context.
    """
    for k_low, k_high, eps in candidate_tests(N, center):
        fpr = _binom_cdf(k_low, N, center) + (1 - _binom_cdf(k_high - 1, N, center))
        fnr = _binom_cdf(k_high - 1, N, signal_strength + center) - _binom_cdf(
            k_low, N, signal_strength + center
        )
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
        if _binom_cdf(n // 2, n, noise_level) < acceptable_misclassification:
            return n
    raise ValueError("not reachable")


#: Chance ``denoise_accept_labels`` moves a label the evidence does not support,
#: and equally the chance it fails to move one it should.
DENOISE_FAILURE_PROB = 1e-5


def _decides(num_samples, signal_strength, boundary, failure_prob):
    """Whether a state either side of the boundary reaches significance at this size.

    ``isf``/``ppf`` invert the tails ``binomial_side_of_boundary`` tests, so
    ``high`` and ``low`` are exactly the counts it calls significant.  An off the
    end of the range gives a zero-probability tail, which decides nothing.
    """
    binom = scipy.stats.binom
    high = binom.isf(failure_prob, num_samples, boundary) + 1
    low = binom.ppf(failure_prob, num_samples, boundary) - 1
    return (
        min(
            binom.sf(high - 1, num_samples, boundary + signal_strength),
            binom.cdf(low, num_samples, boundary - signal_strength),
        )
        >= 1 - failure_prob
    )


def denoise_sample_size(
    signal_strength, boundary=0.5, *, failure_prob=DENOISE_FAILURE_PROB
):
    """Samples one state needs before its own accept rate decides its label.

    A state whose strings are accepted answers at ``boundary + signal_strength``
    and one whose strings are not at ``boundary - signal_strength``, the same
    reading of the boundary the accept-preserving test takes.  Sized so failing
    to decide is as unlikely as deciding wrongly.
    """
    if not 0 <= boundary - signal_strength or not boundary + signal_strength <= 1:
        return None

    def decides(n):
        return _decides(n, signal_strength, boundary, failure_prob)

    n = 1
    while not decides(n):
        n *= 2
    lo, hi = n // 2, n
    while lo < hi:
        mid = (lo + hi) // 2
        if decides(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def binomial_side_of_boundary(num_accepts, num_samples, boundary, *, failure_prob=1e-5):
    """Binomial test of num_accepts/num_samples against accept rate ``boundary``.

    Returns True if the count is significantly above ``boundary``, False if
    significantly below, and None if neither tail clears ``failure_prob`` (including
    small ``num_samples``).
    """
    above = 1 - _binom_cdf(num_accepts - 1, num_samples, boundary)
    if above < failure_prob:
        return True
    below = _binom_cdf(num_accepts, num_samples, boundary)
    if below < failure_prob:
        return False
    return None
