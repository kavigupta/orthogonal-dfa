import functools
import itertools
import math
from typing import Iterator, Optional, Tuple

import numpy as np
import scipy
import scipy.special
import scipy.stats


def binom_cdf(k, n, p):
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


def binom_sf(k, n, p):
    """``1 - binom_cdf(k, n, p)``, without losing the tail below float epsilon."""
    if k < 0:
        return 1.0
    if k >= n:
        return 0.0
    return scipy.special.bdtrc(k, n, p)


def population_size_and_evidence_margin(
    signal_strength, cross_limit, acceptable_fnr, *, center
) -> Tuple[int, float]:
    """
    Decisions will be made by taking N samples and seeing if the proportion is outside
    (center - epsilon, center + epsilon). The true distribution has accept rate
    center + signal_strength and reject rate center - signal_strength.

    We want a rate outside the band to land past its far edge at most cross_limit
    of the time, and FNR (under the true distribution) at most acceptable_fnr.
    """
    assert signal_strength > 0
    # Both class rates have to be probabilities. Otherwise no band ever meets the
    # FNR and the search below doubles N forever rather than failing.
    assert 0 <= center - signal_strength and center + signal_strength <= 1, (
        center,
        signal_strength,
    )
    N_low = 1
    N_high = None
    while N_high is None or N_low < N_high:
        if N_high is None:
            N_try = N_low * 2
        else:
            N_try = (N_low + N_high) // 2
        result = evidence_margin_for_population_size(
            signal_strength, cross_limit, acceptable_fnr, N_try, center=center
        )
        if result is None:
            N_low = N_try + 1
        else:
            N_high = N_try
    res = evidence_margin_for_population_size(
        signal_strength, cross_limit, acceptable_fnr, N_high, center=center
    )
    assert res is not None
    return res


def cross_limit_for_coverage_error(
    signal_strength, acceptable_fnr, max_coverage_error, *, center
):
    """The loosest cross limit whose band holds ``max_coverage_error``.

    A side of the cut keeps its class while the share of it belonging to the other
    stays under ``(1 - eps/signal)/2``, since a prefix on the wrong side reads
    ``2 * signal`` from where the side is held.  So the bound asks the band for a
    width, and width is bought by asking for a smaller limit.

    ``eps`` only grows as the limit falls, so the limits meeting the width are an
    interval from zero and the largest is the one worth finding.
    """
    wanted = signal_strength * (1 - 2 * max_coverage_error)
    if wanted <= 0:
        return 1.0
    low, high = -30.0, 0.0
    for _ in range(24):
        mid = (low + high) / 2
        _, eps = population_size_and_evidence_margin(
            signal_strength, 10**mid, acceptable_fnr, center=center
        )
        low, high = (mid, high) if eps >= wanted else (low, mid)
    # A band this wide leaves the class it is meant to admit inside it, so no
    # limit buys one: the undecided rate has to give first.
    assert low > -30, (
        f"no cross limit holds the cut's error under {max_coverage_error} "
        f"at a signal of {signal_strength} and an undecided rate of {acceptable_fnr}"
    )
    return 10**low


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


#: In a state whose read is undecided at least a third of the time, the most its
#: rarer decided side may be read.
MINORITY_READ_LIMIT = 0.01

#: In such a state, the most its rarer decided side may be read per undecided read.
MINORITY_UNDECIDED_RATIO = 4e-3


def evidence_margin_for_population_size(
    signal_strength, cross_limit, acceptable_fnr, N, *, center
) -> Optional[Tuple[int, float]]:
    """
    See population_size_and_evidence_margin for context.
    """
    for k_low, k_high, eps in candidate_tests(N, center):
        # Crossing only gets less likely further from the band, so the worst rates
        # are its edges: just above k_high - 1, and at k_low.
        cross = max(
            binom_cdf(k_low, N, (k_high - 1) / N),
            binom_sf(k_high - 1, N, k_low / N),
        )
        # Consider the false-negative rate for both elements
        # at margin above and below the center.
        fnr = max(
            binom_cdf(k_high - 1, N, center + side) - binom_cdf(k_low, N, center + side)
            for side in (signal_strength, -signal_strength)
        )
        if (
            cross <= cross_limit
            and fnr <= acceptable_fnr
            and reads_trichotomous(
                k_low,
                k_high,
                N,
                accept_rate=center + signal_strength,
                reject_rate=center - signal_strength,
                limit=cross_limit,
                minority_limit=MINORITY_READ_LIMIT,
                minority_ratio=MINORITY_UNDECIDED_RATIO,
            )
        ):
            return N, eps
    return None


@functools.lru_cache(maxsize=8)
def _vote_parts(N, accept_rate, reject_rate):
    """P(Y = k), P(Z <= k) and P(Z >= k) as three arrays, each indexed `[a, k]` for a
    state from which `a` of the `N` suffixes lead into the language, where Y and Z
    count the accept votes among those `a` suffixes and among the other `N - a`."""
    a = np.arange(N + 1)[:, None]
    count = np.arange(N + 1)[None, :]
    y_pmf = scipy.stats.binom.pmf(count, a, accept_rate)
    z_pmf = scipy.stats.binom.pmf(count, N - a, reject_rate)
    z_le = np.cumsum(z_pmf, axis=1)
    z_ge = np.cumsum(z_pmf[:, ::-1], axis=1)[:, ::-1]
    return y_pmf, z_le, z_ge


def reads_trichotomous(
    k_low, k_high, N, *, accept_rate, reject_rate, limit, minority_limit, minority_ratio
):
    """Whether, for every `a`, the read of a state from which `a` of the `N` suffixes
    lead into the language (each voting accept at `accept_rate`, the rest at
    `reject_rate`) is accept at most `limit` of the time, or reject at most `limit`,
    or undecided at least a third with accept or reject at most `minority_limit` and the
    rarer of them at most `minority_ratio` times the undecided.
    `BandPasses` in proofs/OrthoDFA/FamilyRead.lean.

    `evidence_margin_for_population_size`'s other two criteria do not imply it: a mean
    just inside the band at a small count can sit at or below `k_low` more than 2/3 of
    the time while its far tail is a hair above the band edge's.
    """
    y_pmf, z_le, z_ge = _vote_parts(N, accept_rate, reject_rate)
    count = np.arange(N + 1)
    reject = (y_pmf[:, : k_low + 1] * z_le[:, k_low::-1]).sum(axis=1)
    accept = (y_pmf * z_ge[:, np.maximum(k_high - count, 0)]).sum(axis=1)
    undecided = 1 - accept - reject
    leans = (accept <= minority_limit) | (reject <= minority_limit)
    rare = np.minimum(accept, reject) <= minority_ratio * undecided
    return bool(
        np.all(
            (accept <= limit)
            | (reject <= limit)
            | (leans & (undecided >= 1 / 3) & rare)
        )
    )


def compute_suffix_size_counterexample_gen(acceptable_misclassification, noise_level):
    """
    Computes the suffix size to use for counterexample generation.
    This is an alias for compute_suffix_size_for_counterexample_generation
    to match the naming convention of other hyperparameter generators.
    """
    for n in itertools.count(start=1):
        if binom_cdf(n // 2, n, noise_level) < acceptable_misclassification:
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
    above = 1 - binom_cdf(num_accepts - 1, num_samples, boundary)
    if above < failure_prob:
        return True
    below = binom_cdf(num_accepts, num_samples, boundary)
    if below < failure_prob:
        return False
    return None
